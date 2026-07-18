//! YaneuraOu SFNN ビルド向け nn.bin 保存フォーマット。
//!
//! `bulletou --eval-type SFNN_*` で学習した重みを
//! やねうら王 (`YANEURAOU_ENGINE_SFNN_*`) が `EvalDir` で load
//! できるバイナリに変換するための [`SavedFormat`] 列を組み立てる。
//!
//! 参考にしているやねうら王ソース:
//! - `source/eval/nnue/evaluate_nnue.cpp` (header + 全体 WriteParameters)
//! - `source/eval/nnue/nnue_feature_transformer.h` (FT, LEB128 経路)
//! - `source/eval/nnue/nnue_common.h` (LEB128 magic / format)
//! - `source/eval/nnue/architectures/sfnnwop-1536.h` (Network 構造、各 hash 値)
//! - `source/eval/nnue/layers/affine_transform_*.h` (per-layer i32 bias + i8 weight)
//!
//! バイナリ全体構造:
//! ```text
//! [Header]
//!   u32 LE: NNUE_VERSION = 0x7AF32F16    (やねうら王 kVersion、不一致で load 失敗)
//!   u32 LE: kHashValue                   (任意、やねうら王側は warning のみ)
//!   u32 LE: desc_len
//!   bytes : architecture description (UTF-8、表示専用)
//!
//! [FeatureTransformer]
//!   u32 LE: FT hash = 0x5f134ab8         (SFNN 固定値、`feature_transformer.h:164`)
//!   [LEB128 block: biases]               (USE_ELEMENT_WISE_MULTIPLY 経路、SFNN 必須)
//!     17 bytes: "COMPRESSED_LEB128"
//!     u32 LE  : payload size
//!     bytes   : signed-LEB128 encoded i16 × ft_size
//!   [LEB128 block: weights]
//!     17 bytes: "COMPRESSED_LEB128"
//!     u32 LE  : payload size
//!     bytes   : signed-LEB128 encoded i16 × (ft_size × input_size)
//!
//! [× num_stacks LayerStacks]
//!   u32 LE: Network hash = 0x6333718A    (SFNN 固定値、`sfnnwop-1536.h:54`)
//!   fc_0 biases  : i32 LE × (l1_hidden + 1)               (scale = qa × qb = 8128)
//!   fc_0 weights : i8 × (l1_hidden + 1) × pad32(ft_size)  (row-major、scale = qb = 64)
//!   fc_1 biases  : i32 LE × l2_size                       (scale = qa × qb)
//!   fc_1 weights : i8 × l2_size × pad32(l1_hidden × 2)    (row-major、scale = qb)
//!   fc_2 biases  : i32 LE × 1                             (scale = qa × qb)
//!   fc_2 weights : i8 × 1 × pad32(l2_size)                (row-major、scale = qb)
//! ```
//!
//! やねうら王側の SIMD-friendly chunked layout (`get_weight_index_scrambled`)
//! はエンジン読み込み時に reader 内部で適用されるため、本実装ではファイルを
//! **行優先 `(out, in)`** で書けば十分。padding 領域 (in >= in_dim_real) は 0 埋め。

use std::rc::Rc;

use crate::trainer::save::SavedFormat;
use crate::value::nnue_save::NnueFeatureSet;
use bullet_trainer::model::save::ModelWeights;

/// SFNN-1536 NNUE quantisation scales (`bullet-shogi` 由来、Stockfish の標準値)。
pub const QA: i16 = 127;
/// `fc_*` 層の i8 weight scale。`fc_*` 層の i32 bias scale は `QA × QB` (= 8128)。
pub const QB: i16 = 64;

/// やねうら王 `kVersion` (`nnue_common.h:62`)。不一致だと `FileMismatch` で load 失敗。
pub const NNUE_VERSION: u32 = 0x7AF32F16;
/// やねうら王 SFNN の FT 固定 hash (`nnue_feature_transformer.h:164`)。
pub const FT_HASH_SFNN: u32 = 0x5F134AB8;
/// やねうら王 SFNN の Network 固定 hash (`sfnnwop-1536.h:54`)。
pub const NETWORK_HASH_SFNN: u32 = 0x6333718A;
/// やねうら王 SFNN の wide kHashValue (`evaluate_nnue.h:25`)。warning のみで参照。
pub const KHASH_SFNN: u32 = 0x3C203B32;
/// LEB128 圧縮ブロック先頭の magic (`nnue_common.h:65`)。null terminator は含まない。
pub const LEB128_MAGIC: &[u8] = b"COMPRESSED_LEB128";

/// SFNN / LayerStacks 用 nn.bin save format ビルダパラメータ。
#[derive(Debug, Clone, Copy)]
pub struct Sfnn1536SaveParams {
    /// SFNN 系の `NnueFeatureSet`。description 文字列に
    /// 表示名 (例 `HalfKA_hm2(Friend)`) を埋め込むために使う。
    pub feature_set: NnueFeatureSet,
    /// 入力特徴量次元 (= `feature_set.input_size()`)。
    pub input_size: usize,
    /// Feature Transformer 出力次元 (= `kTransformedFeatureDimensions`)。
    pub ft_size: usize,
    /// 隠れ層 1 の実効次元 (`kHidden1Dims`)。+1 PSQT
    /// shortcut neuron は内部で自動付加。
    pub l1_hidden: usize,
    /// 隠れ層 2 次元 (`kHidden2Dims`)。
    pub l2_size: usize,
    /// LayerStacks 数。
    pub num_stacks: usize,
    /// L1 に nnue-pytorch-style factorized shared term (`l1f`) があるか。
    /// true の場合、保存時に `l1f` を各 bucket の `fc_0` に fold する。
    pub factorized_l1: bool,
}

impl Sfnn1536SaveParams {
    /// やねうら王 1536x2-15-32 ビルドと完全互換のデフォルト値。
    pub fn yaneuraou_default(feature_set: NnueFeatureSet) -> Self {
        Self {
            feature_set,
            input_size: feature_set.input_size(),
            ft_size: 1536,
            l1_hidden: 15,
            l2_size: 32,
            num_stacks: 9,
            factorized_l1: false,
        }
    }

    /// `fc_0` の出力次元 (= 15 hidden + 1 PSQT shortcut)。
    fn l1_out(&self) -> usize {
        self.l1_hidden + 1
    }

    /// `fc_1` の入力次元 (= `[SqrCReLU; CReLU]` concat の 30)。
    fn l2_in(&self) -> usize {
        self.l1_hidden * 2
    }

    /// やねうら王 `GetArchitectureString()` 互換の description 文字列。
    ///
    /// 表示専用 (parse されない) だが、`bench` 時の log に出るので可読性のため
    /// 形式は揃えておく。
    fn architecture_string(&self) -> String {
        // やねうら王 evaluate_nnue.cpp:161-165:
        //   "ModelType=SFNNWithoutPsqt;Features=...,Network=...{LayerStack=N}"
        // FT::GetStructureString = "<feature>[<in>-><ft>x2]"
        // Network::GetStructureString = "SFNN-<ft>" (sfnnwop-1536.h:58)
        format!(
            "ModelType=SFNNWithoutPsqt;Features={}[{}->{}x2],Network=SFNN-{}{{LayerStack={}}}",
            self.feature_set.display_name(),
            self.input_size,
            self.ft_size,
            self.ft_size,
            self.num_stacks
        )
    }
}

/// 32 byte alignment への切り上げ。やねうら王の `kPaddedInputDimensions` 相当。
fn pad32(n: usize) -> usize {
    n.div_ceil(32) * 32
}

/// 1 個の `i16` を signed-LEB128 にエンコードして bytes に push する。
///
/// やねうら王 `nnue_common.h:read_leb_128` と対称。各 byte の下位 7 bit がデータ、
/// MSB が continuation flag。最後の byte の 0x40 bit が sign bit (符号拡張用)。
fn push_signed_leb128_i16(out: &mut Vec<u8>, value: i16) {
    let mut v = i32::from(value); // 算術右シフトを使うため i32 に拡張
    let bits = 16;
    let mut more = true;
    while more {
        let byte = (v as u8) & 0x7F;
        v >>= 7; // 算術右シフトで符号維持
        let sign_bit = byte & 0x40 != 0;
        let _ = bits; // 参考: i16 だと 3 byte で必ず収まる
        if (v == 0 && !sign_bit) || (v == -1 && sign_bit) {
            out.push(byte);
            more = false;
        } else {
            out.push(byte | 0x80);
        }
    }
}

/// LEB128 ブロック全体 (magic + size + payload) を組み立てる。
fn make_leb128_block(values: &[i16]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(values.len() * 2);
    for &v in values {
        push_signed_leb128_i16(&mut payload, v);
    }
    let mut out = Vec::with_capacity(LEB128_MAGIC.len() + 4 + payload.len());
    out.extend_from_slice(LEB128_MAGIC);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend(payload);
    out
}

/// バイト列を `SavedFormat` 経由で書き出すためのパススルー変換。
/// `(b as i8) as f32` で各 byte を f32 として渡し、`quantise::<i8>(1)` で
/// `f32 × 1.0 → i8 → 1 byte` のラウンドトリップを取る。
fn bytes_to_f32_passthrough(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&b| (b as i8) as f32).collect()
}

/// 量子化ヘルパ: f32 → i16 (FT、scale = qa)。範囲外は clamp。
fn quantise_i16(value: f32, scale: f32) -> i16 {
    let q = (f64::from(value) * f64::from(scale)).round();
    q.clamp(i16::MIN as f64, i16::MAX as f64) as i16
}

/// 量子化ヘルパ: f32 → i32 (fc bias、scale = qa × qb)。範囲外は clamp。
fn quantise_i32(value: f32, scale: f32) -> i32 {
    let q = (f64::from(value) * f64::from(scale)).round();
    q.clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

/// 量子化ヘルパ: f32 → i8 (fc weight、scale = qb)。範囲外は clamp。
fn quantise_i8(value: f32, scale: f32) -> i8 {
    let q = (f64::from(value) * f64::from(scale)).round();
    q.clamp(-128.0, 127.0) as i8
}

/// ModelWeights から f32 配列を取り出す。`TValue::F32` 以外なら panic。
fn read_f32(graph: &ModelWeights, id: &str) -> Rc<Vec<f32>> {
    use bullet_compiler::tensor::TValue;
    match graph.get(id).values {
        TValue::F32(v) => Rc::new(v),
        _ => panic!("expected F32 weights for `{id}`"),
    }
}

// =============================================================================
// メインビルダ
// =============================================================================

/// SFNN-1536 / LayerStacks=N の nn.bin を書く `SavedFormat` 列を構築する。
///
/// 列は appender 順に評価され、各 `SavedFormat` が生成した bytes を nn.bin に
/// そのまま追記する。
pub fn build_sfnn_1536_save_format(params: Sfnn1536SaveParams) -> Vec<SavedFormat> {
    let mut formats: Vec<SavedFormat> = Vec::new();

    // ---- Header ----
    {
        let arch = params.architecture_string();
        let arch_bytes = arch.as_bytes();
        let mut header = Vec::with_capacity(12 + arch_bytes.len());
        header.extend_from_slice(&NNUE_VERSION.to_le_bytes());
        header.extend_from_slice(&KHASH_SFNN.to_le_bytes());
        header.extend_from_slice(&(arch_bytes.len() as u32).to_le_bytes());
        header.extend_from_slice(arch_bytes);
        formats.push(SavedFormat::custom(header));
    }

    // ---- FeatureTransformer ----
    formats.push(SavedFormat::custom(FT_HASH_SFNN.to_le_bytes().to_vec()));

    // FT biases: LEB128 block of i16 × ft_size
    formats.push(
        SavedFormat::empty()
            .transform(move |graph, _| {
                let l0b = read_f32(graph, "l0b");
                let qa_f = QA as f32;
                let q: Vec<i16> = l0b.iter().map(|&v| quantise_i16(v, qa_f)).collect();
                let block = make_leb128_block(&q);
                bytes_to_f32_passthrough(&block)
            })
            .quantise::<i8>(1),
    );

    // FT weights: LEB128 block of i16 × (ft_size × input_size)
    formats.push(
        SavedFormat::empty()
            .transform(move |graph, _| {
                let l0w = read_f32(graph, "l0w");
                let qa_f = QA as f32;
                let q: Vec<i16> = l0w.iter().map(|&v| quantise_i16(v, qa_f)).collect();
                let block = make_leb128_block(&q);
                bytes_to_f32_passthrough(&block)
            })
            .quantise::<i8>(1),
    );

    // ---- LayerStacks ----
    let l1_out = params.l1_out();
    let l2_in = params.l2_in();
    let ft_size = params.ft_size;
    let l2_size = params.l2_size;
    let num_stacks = params.num_stacks;
    let factorized_l1 = params.factorized_l1;

    for k in 0..num_stacks {
        // Network hash (per-stack header u32 LE).
        formats.push(SavedFormat::custom(NETWORK_HASH_SFNN.to_le_bytes().to_vec()));

        // fc_0: bias i32 × l1_out, weight i8 × l1_out × pad32(ft_size).
        let stack = k;
        formats.push(
            SavedFormat::empty()
                .transform(move |graph, _| {
                    let l1b = read_f32(graph, "l1b");
                    let l1fb = factorized_l1.then(|| read_f32(graph, "l1fb"));
                    let bias_scale = (QA as i32 * QB as i32) as f32;
                    let mut bytes = Vec::with_capacity(l1_out * 4);
                    for o in 0..l1_out {
                        let mut v = l1b[stack * l1_out + o];
                        if let Some(l1fb) = l1fb.as_ref() {
                            v += l1fb[o];
                        }
                        bytes.extend_from_slice(&quantise_i32(v, bias_scale).to_le_bytes());
                    }
                    bytes_to_f32_passthrough(&bytes)
                })
                .quantise::<i8>(1),
        );

        formats.push(
            SavedFormat::empty()
                .transform(move |graph, _| {
                    let l1w = read_f32(graph, "l1w"); // shape (ns*l1_out, ft_size), index `feat*ns*l1_out + stack*l1_out + o`
                    let l1fw = factorized_l1.then(|| read_f32(graph, "l1fw"));
                    let pad_in = pad32(ft_size);
                    let w_scale = QB as f32;
                    let mut bytes = Vec::with_capacity(l1_out * pad_in);
                    for o in 0..l1_out {
                        for in_ in 0..pad_in {
                            let q = if in_ < ft_size {
                                let mut v = l1w[in_ * (num_stacks * l1_out) + stack * l1_out + o];
                                if let Some(l1fw) = l1fw.as_ref() {
                                    v += l1fw[in_ * l1_out + o];
                                }
                                quantise_i8(v, w_scale)
                            } else {
                                0
                            };
                            bytes.push(q as u8);
                        }
                    }
                    bytes_to_f32_passthrough(&bytes)
                })
                .quantise::<i8>(1),
        );

        // fc_1: bias i32 × l2_size, weight i8 × l2_size × pad32(l2_in).
        formats.push(
            SavedFormat::empty()
                .transform(move |graph, _| {
                    let l2b = read_f32(graph, "l2b"); // shape (ns*l2_size, 1)
                    let bias_scale = (QA as i32 * QB as i32) as f32;
                    let mut bytes = Vec::with_capacity(l2_size * 4);
                    for o in 0..l2_size {
                        bytes.extend_from_slice(&quantise_i32(l2b[stack * l2_size + o], bias_scale).to_le_bytes());
                    }
                    bytes_to_f32_passthrough(&bytes)
                })
                .quantise::<i8>(1),
        );

        formats.push(
            SavedFormat::empty()
                .transform(move |graph, _| {
                    let l2w = read_f32(graph, "l2w"); // shape (ns*l2_size, l2_in), index `feat*ns*l2_size + stack*l2_size + o`
                    let pad_in = pad32(l2_in);
                    let w_scale = QB as f32;
                    let mut bytes = Vec::with_capacity(l2_size * pad_in);
                    for o in 0..l2_size {
                        for in_ in 0..pad_in {
                            let q = if in_ < l2_in {
                                quantise_i8(l2w[in_ * (num_stacks * l2_size) + stack * l2_size + o], w_scale)
                            } else {
                                0
                            };
                            bytes.push(q as u8);
                        }
                    }
                    bytes_to_f32_passthrough(&bytes)
                })
                .quantise::<i8>(1),
        );

        // fc_2: bias i32 × 1, weight i8 × 1 × pad32(l2_size).
        formats.push(
            SavedFormat::empty()
                .transform(move |graph, _| {
                    let l3b = read_f32(graph, "l3b"); // shape (ns, 1)
                    let bias_scale = (QA as i32 * QB as i32) as f32;
                    let bytes = quantise_i32(l3b[stack], bias_scale).to_le_bytes().to_vec();
                    bytes_to_f32_passthrough(&bytes)
                })
                .quantise::<i8>(1),
        );

        formats.push(
            SavedFormat::empty()
                .transform(move |graph, _| {
                    let l3w = read_f32(graph, "l3w"); // shape (ns, l2_size), index `feat*ns + stack`
                    let pad_in = pad32(l2_size);
                    let w_scale = QB as f32;
                    let mut bytes = Vec::with_capacity(pad_in);
                    for in_ in 0..pad_in {
                        let q = if in_ < l2_size { quantise_i8(l3w[in_ * num_stacks + stack], w_scale) } else { 0 };
                        bytes.push(q as u8);
                    }
                    bytes_to_f32_passthrough(&bytes)
                })
                .quantise::<i8>(1),
        );
    }

    formats
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad32_basic() {
        assert_eq!(pad32(0), 0);
        assert_eq!(pad32(1), 32);
        assert_eq!(pad32(30), 32);
        assert_eq!(pad32(32), 32);
        assert_eq!(pad32(33), 64);
        assert_eq!(pad32(1536), 1536);
    }

    #[test]
    fn leb128_roundtrip_small_positives() {
        for v in 0..=127 {
            let mut buf = Vec::new();
            push_signed_leb128_i16(&mut buf, v);
            // signed-LEB128 で 0..=63 は 1 byte、64..=127 は 2 byte
            // (0x40 が立つと「次に進む」と誤解されないために 0x00 を追加)
            if v <= 63 {
                assert_eq!(buf.len(), 1, "v={v}: {:?}", buf);
                assert_eq!(buf[0], v as u8);
            } else {
                assert_eq!(buf.len(), 2, "v={v}: {:?}", buf);
                assert_eq!(buf[0], (v as u8) | 0x80);
                assert_eq!(buf[1], 0x00);
            }
        }
    }

    #[test]
    fn leb128_roundtrip_small_negatives() {
        // -1 は 1 byte 0x7f
        let mut buf = Vec::new();
        push_signed_leb128_i16(&mut buf, -1);
        assert_eq!(buf, vec![0x7f]);

        // -64 は 1 byte 0x40 (= 7 bit 0x40、sign-bit 立ち、続行なし)
        let mut buf = Vec::new();
        push_signed_leb128_i16(&mut buf, -64);
        assert_eq!(buf, vec![0x40]);

        // -65 は 2 byte: 0xff (= -65 の下位 7 bit = 0x7f) | 0x80 = 0xff, 続いて 0x7e (符号拡張で停止)
        // 計算: -65 = ...111111110111111  下位 7 bit = 0x3f、MSB 立てて 0xbf
        // 右シフト 7: -1 (算術右シフト)、-1 を 7 bit にすると 0x7f、sign bit (0x40) 立つ
        // → 2 byte: [0xbf, 0x7f]
        let mut buf = Vec::new();
        push_signed_leb128_i16(&mut buf, -65);
        assert_eq!(buf, vec![0xbf, 0x7f]);
    }

    #[test]
    fn leb128_block_layout() {
        let block = make_leb128_block(&[0, 1, -1, 127]);
        // magic + u32 size + payload
        assert_eq!(&block[..17], b"COMPRESSED_LEB128");
        let size = u32::from_le_bytes(block[17..21].try_into().unwrap());
        assert_eq!(size as usize, block.len() - 21);
        // payload: 0x00, 0x01, 0x7f, 0xff, 0x00 (127 needs 2 bytes)
        assert_eq!(&block[21..], &[0x00, 0x01, 0x7f, 0xff, 0x00]);
    }

    #[test]
    fn header_bytes_layout() {
        let params = Sfnn1536SaveParams::yaneuraou_default(NnueFeatureSet::HalfKaHm2);
        let formats = build_sfnn_1536_save_format(params);
        assert!(!formats.is_empty());
        // First SavedFormat is the header (custom bytes).
        assert!(formats[0].is_custom(), "first SavedFormat must be the header custom bytes");
    }

    #[test]
    fn formats_count_matches_layout() {
        // 1 (header) + 1 (FT hash) + 2 (FT LEB128 blocks) + N × 7 (per stack:
        // hash + fc_0 bias/weights + fc_1 bias/weights + fc_2 bias/weights)
        let params = Sfnn1536SaveParams::yaneuraou_default(NnueFeatureSet::HalfKaHm2);
        let formats = build_sfnn_1536_save_format(params);
        let expected = 1 + 1 + 2 + 9 * 7;
        assert_eq!(formats.len(), expected, "expected {expected} SavedFormat entries");
    }

    #[test]
    fn architecture_string_matches_yaneuraou_form() {
        let params = Sfnn1536SaveParams::yaneuraou_default(NnueFeatureSet::HalfKaHm2);
        let s = params.architecture_string();
        assert_eq!(
            s,
            "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536{LayerStack=9}"
        );
    }

    #[test]
    fn quantise_i8_clamps() {
        // QB=64 で f32 1.5 → 96; 3.0 → clamp to 127 (192 > 127)
        assert_eq!(quantise_i8(1.5, QB as f32), 96);
        assert_eq!(quantise_i8(3.0, QB as f32), 127);
        assert_eq!(quantise_i8(-3.0, QB as f32), -128);
    }

    #[test]
    fn quantise_i16_clamps() {
        assert_eq!(quantise_i16(0.0, QA as f32), 0);
        assert_eq!(quantise_i16(1.0, QA as f32), 127);
        // 300 * 127 = 38100 > i16::MAX (32767) → clamp
        assert_eq!(quantise_i16(300.0, QA as f32), i16::MAX);
    }
}
