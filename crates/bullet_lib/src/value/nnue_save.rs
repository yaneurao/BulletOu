//! YaneuraOu / Stockfish (nnue-pytorch) 互換の NNUE バイナリヘッダー生成と
//! 量子化向け重みパディングのヘルパー。
//!
//! `quantised.bin` を NNUE 形式で書くために `SavedFormat::custom(...)` で
//! 差し込むヘッダーバイト列、Feature Transformer / Network レイヤーハッシュ、
//! description 文字列、L1 bias scale、SIMD 32 バイトアライメント用の重み
//! パディングを提供する。各 NNUE バリアント (HalfKP / HalfKA / HalfKA_hm) の
//! 学習スクリプトから共通に呼ばれることを想定。
//!
//! 参考実装: `examples/shogi_simple.rs` の `OutputFormat::Standard` 経路。
//! ここに切り出すことで `bulletou` などの統合 CLI からも再利用できる。

use crate::game::inputs::{
    FEATURE_HASH, FEATURE_HASH_HALFKPE9, FEATURE_HASH_HM_V2, FEATURE_HASH_KP, FEATURE_HASH_NONMIRROR,
};

/// YaneuraOu / Stockfish 互換 NNUE バイナリのバージョンマジック。
pub const NNUE_VERSION: u32 = 0x7AF32F16;

/// 対応する NNUE feature set。`feature_hash`, `input_size`, description
/// 内の表示名はこの enum を経由して引く。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NnueFeatureSet {
    /// HalfKP(Friend) — 玉 × 駒 sparse 特徴量 (125,388 次元)。
    HalfKp,
    /// HalfKA(Friend) — 玉も特徴量に含めた 81 bucket 版 (138,510 次元)。
    HalfKa,
    /// HalfKA_hm(Friend) — half-mirror 化した HalfKA (45 bucket, 73,305 次元)。
    HalfKaHm,
    /// K-P(Friend) — YaneuraOu の `FeatureSet<K, P>` (`k-p_256x2-32-32.h`)。
    /// K (玉 2 個, 162 次元) + P (玉以外, 1548 次元) = 1710 次元 / perspective。
    Kp,
    /// HalfKPE9(Friend) — YaneuraOu の `halfkpe9_*` 系。HalfKP の
    /// `(king_sq, piece)` ペアに、その駒のマスの利き数情報を
    /// 9 通り (= 自軍 0/1/2 × 敵軍 0/1/2) ぶん多重化。
    /// 1,128,492 次元 / perspective。
    HalfKpe9,
}

impl NnueFeatureSet {
    /// nnue-pytorch / YaneuraOu 互換の feature hash。
    pub fn feature_hash(self) -> u32 {
        match self {
            NnueFeatureSet::HalfKp => FEATURE_HASH,
            NnueFeatureSet::HalfKa => FEATURE_HASH_NONMIRROR,
            NnueFeatureSet::HalfKaHm => FEATURE_HASH_HM_V2,
            NnueFeatureSet::Kp => FEATURE_HASH_KP,
            NnueFeatureSet::HalfKpe9 => FEATURE_HASH_HALFKPE9,
        }
    }

    /// description 文字列に埋め込む入力次元 (= 特徴量数)。
    pub fn input_size(self) -> usize {
        match self {
            NnueFeatureSet::HalfKp => 125_388,
            NnueFeatureSet::HalfKa => 138_510,
            NnueFeatureSet::HalfKaHm => 73_305,
            NnueFeatureSet::Kp => 1_710,
            NnueFeatureSet::HalfKpe9 => 1_128_492,
        }
    }

    /// description 文字列の先頭に出る表示名。
    pub fn display_name(self) -> &'static str {
        match self {
            NnueFeatureSet::HalfKp => "HalfKP(Friend)",
            NnueFeatureSet::HalfKa => "HalfKA(Friend)",
            NnueFeatureSet::HalfKaHm => "HalfKA_hm(Friend)",
            // FeatureSet<K, P>::GetName() = "K+P" (feature_set.h の "+" 結合) だが、
            // nnue-pytorch / YaneuraOu の description では (Friend) suffix を付ける
            // 慣習に従って "K-P(Friend)" 表記とする。engine 側が description を
            // 厳密検証しない限り、network_hash が一致すれば load 可能。
            NnueFeatureSet::Kp => "K-P(Friend)",
            // YaneuraOu features/half_kpe9.h の kName と一致 (大文字 "E")。
            NnueFeatureSet::HalfKpe9 => "HalfKPE9(Friend)",
        }
    }
}

/// 学習時の活性化関数 (量子化スケール計算で使う)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Crelu,
    Screlu,
}

/// 32 バイトアライメント (SIMD 用パディング後の入力次元) を返す。
pub fn pad32(size: usize) -> usize {
    size.div_ceil(32) * 32
}

/// nnue-pytorch 互換の `fc_hash` を計算する。
///
/// - InputSlice base: `0xEC42E90D`
/// - Layer base:      `0xCC03DAE4`
/// - ClippedReLU:     `0x538D24C7`
///
/// `[l2_size, l3_size, 1]` の 3 fully-connected 層を順に巻き込み、output 層
/// 以外には ClippedReLU hash を加算する。
pub fn compute_fc_hash(l1_size: usize, l2_size: usize, l3_size: usize) -> u32 {
    let mut prev_hash: u32 = 0xEC42E90D;
    prev_hash ^= (l1_size * 2) as u32;

    let layer_sizes = [l2_size, l3_size, 1usize];
    for (i, &out_features) in layer_sizes.iter().enumerate() {
        let mut layer_hash: u32 = 0xCC03DAE4;
        layer_hash = layer_hash.wrapping_add(out_features as u32);
        layer_hash ^= prev_hash >> 1;
        layer_hash ^= prev_hash << 31;
        if i < 2 {
            layer_hash = layer_hash.wrapping_add(0x538D24C7);
        }
        prev_hash = layer_hash;
    }
    prev_hash
}

/// Feature Transformer のレイヤーハッシュ = `feature_hash XOR (l1_size * 2)`。
pub fn ft_hash(feature_set: NnueFeatureSet, l1_size: usize) -> u32 {
    feature_set.feature_hash() ^ ((l1_size * 2) as u32)
}

/// nnue-pytorch / YaneuraOu 互換の network hash
/// = `fc_hash XOR feature_hash XOR (l1_size * 2)`。
pub fn compute_network_hash(feature_set: NnueFeatureSet, l1_size: usize, l2_size: usize, l3_size: usize) -> u32 {
    compute_fc_hash(l1_size, l2_size, l3_size) ^ feature_set.feature_hash() ^ ((l1_size * 2) as u32)
}

/// nnue-pytorch 形式の description 文字列。
pub fn description(feature_set: NnueFeatureSet, l1_size: usize, l2_size: usize, l3_size: usize) -> String {
    format!(
        "Features={}[{}->{}x2],Network=AffineTransform[1<-{}](ClippedReLU[{}](AffineTransform[{}<-{}](ClippedReLU[{}](AffineTransformSparseInput[{}<-{}](InputSlice[{}(0:{})])))))",
        feature_set.display_name(),
        feature_set.input_size(),
        l1_size,
        l3_size,
        l3_size,
        l3_size,
        l2_size,
        l2_size,
        l2_size,
        l1_size * 2,
        l1_size * 2,
        l1_size * 2,
    )
}

/// NNUE バイナリのヘッダー = `NNUE_VERSION` + `network_hash` + `desc_len` + `description`。
/// `SavedFormat::custom(...)` の引数にそのまま渡せるバイト列を返す。
pub fn header_bytes(feature_set: NnueFeatureSet, l1_size: usize, l2_size: usize, l3_size: usize) -> Vec<u8> {
    let network_hash = compute_network_hash(feature_set, l1_size, l2_size, l3_size);
    let desc = description(feature_set, l1_size, l2_size, l3_size);
    let desc_bytes = desc.as_bytes();
    let mut header = Vec::with_capacity(12 + desc_bytes.len());
    header.extend_from_slice(&NNUE_VERSION.to_le_bytes());
    header.extend_from_slice(&network_hash.to_le_bytes());
    header.extend_from_slice(&(desc_bytes.len() as u32).to_le_bytes());
    header.extend_from_slice(desc_bytes);
    header
}

/// `ft_hash` の little-endian 4 バイト表現。
pub fn ft_hash_bytes(feature_set: NnueFeatureSet, l1_size: usize) -> Vec<u8> {
    ft_hash(feature_set, l1_size).to_le_bytes().to_vec()
}

/// `fc_hash` の little-endian 4 バイト表現 (= network layer hash)。
pub fn network_layer_hash_bytes(l1_size: usize, l2_size: usize, l3_size: usize) -> Vec<u8> {
    compute_fc_hash(l1_size, l2_size, l3_size).to_le_bytes().to_vec()
}

/// L1 層の bias 量子化スケール。
///
/// L1 層の入力スケールは活性化関数の出力スケールに依存する:
///
/// | 活性化     | qa  | 出力スケール | L1 bias scale |
/// |------------|-----|--------------|----------------|
/// | CReLU      | 127 | 127          | 127 × qb       |
/// | CReLU      | 255 | 255          | 255 × qb       |
/// | SCReLU 255 | 255 | 127 (x²>>9)  | 127 × qb       |
/// | Pairwise   | 255 | 127 (ab>>9)  | 127 × qb       |
///
/// SCReLU / Pairwise は QA=255 でも出力が 127 にスケールダウンされる点に注意。
pub fn l1_bias_scale(activation: Activation, pairwise: bool, qa: i16, qb: i16) -> i32 {
    if pairwise {
        let qa_i32 = i32::from(qa);
        let shift = if qa >= 255 { 9 } else { 7 };
        ((qa_i32 * qa_i32) >> shift) * i32::from(qb)
    } else if matches!(activation, Activation::Screlu) && qa >= 255 {
        127 * i32::from(qb)
    } else {
        i32::from(qa) * i32::from(qb)
    }
}

/// 32 バイトアライメントのために入力次元を pad32() 倍に拡張した重みを返す。
/// 入力次元がすでに 32 の倍数の場合はコピーを返すだけ。
///
/// 重みは row-major (`out_dim × in_dim`) を仮定する。
pub fn pad_weights_for_simd(weights: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let padded_in_dim = pad32(in_dim);
    if padded_in_dim == in_dim {
        return weights.to_vec();
    }
    let mut result = vec![0.0f32; out_dim * padded_in_dim];
    for o in 0..out_dim {
        for i in 0..in_dim {
            result[o * padded_in_dim + i] = weights[o * in_dim + i];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad32_rounds_up() {
        assert_eq!(pad32(0), 0);
        assert_eq!(pad32(1), 32);
        assert_eq!(pad32(32), 32);
        assert_eq!(pad32(33), 64);
        assert_eq!(pad32(127), 128);
    }

    #[test]
    fn header_length_matches_components() {
        let h = header_bytes(NnueFeatureSet::HalfKp, 256, 32, 32);
        // 4 + 4 + 4 + desc_len
        let desc_len = description(NnueFeatureSet::HalfKp, 256, 32, 32).len();
        assert_eq!(h.len(), 12 + desc_len);
        assert_eq!(&h[0..4], NNUE_VERSION.to_le_bytes().as_ref());
    }

    #[test]
    fn pad_weights_no_op_when_aligned() {
        let w = vec![1.0f32, 2.0, 3.0, 4.0];
        let padded = pad_weights_for_simd(&w, 2, 2);
        // in_dim=2 < 32, pad to 32; out_dim=2 -> total 2*32=64
        assert_eq!(padded.len(), 64);
        assert_eq!(padded[0], 1.0);
        assert_eq!(padded[1], 2.0);
        assert_eq!(padded[32], 3.0);
        assert_eq!(padded[33], 4.0);
    }

    #[test]
    fn l1_bias_scale_crelu_qa127() {
        let s = l1_bias_scale(Activation::Crelu, false, 127, 64);
        assert_eq!(s, 127 * 64);
    }

    #[test]
    fn l1_bias_scale_screlu_qa255() {
        let s = l1_bias_scale(Activation::Screlu, false, 255, 64);
        assert_eq!(s, 127 * 64);
    }
}
