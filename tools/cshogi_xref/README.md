# tools/cshogi_xref

BulletOu の HCPE デコード経路 (`MiniPosition::from_hcp` + `to_packed_sfen_value`) が、cshogi の独立実装 (`Board.set_hcp` + `Board.to_psfen`) と **40 byte レベルで一致する**ことを確認するための補助スクリプト。

## hcpe_to_psv.py

cshogi 経由で `.hcpe` を `.psv` (PackedSfenValue, 40 byte/record) に変換する。

依存:

```bash
pip install cshogi numpy
```

使い方:

```bash
# 先頭 10000 件だけ変換 (cross-validate テスト用)
python3 tools/cshogi_xref/hcpe_to_psv.py \
    inbox/ref/sp_dr2-15K_20240210.hcpe \
    inbox/ref/sp_dr2-15K_20240210.psv \
    10000

# 全件 (count を省略)
python3 tools/cshogi_xref/hcpe_to_psv.py \
    inbox/ref/full.hcpe \
    inbox/ref/full.psv
```

## Rust 側のクロスバリデーションテスト

`crates/bullet_lib/src/value/loader/hcpe.rs::tests::cross_validate_against_cshogi_psv`。`#[ignore]` で常時実行はしない。

```bash
# psv を作ってから
python3 tools/cshogi_xref/hcpe_to_psv.py \
    inbox/ref/sp_dr2-15K_20240210.hcpe \
    inbox/ref/sp_dr2-15K_20240210.psv 10000

# Rust テストで照合
cargo test -p bullet_lib --lib cross_validate_against_cshogi_psv -- --ignored --nocapture
```

期待される出力:

```
comparing 10000 records (hcpe total = 4583825, psv total = 10000)
OK: all 10000 records match byte-for-byte (BulletOu == cshogi)
```
