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

## hcpe3_to_psv.py

cshogi の `Board.set_hcp` + `Board.push_move16` + `Board.to_psfen` を順に使い、`.hcpe3` のゲーム単位レコードを ply 単位の psv (PackedSfenValue) に展開して書き出す。

MoveVisits (policy teacher) は読み飛ばす (value 用)。`gamePly` フィールドは ply 番号 (0-indexed) を入れる (BulletOu の `MiniPosition.game_ply` 進行と一致)。

```bash
# 先頭 10000 局面分だけ展開
python3 tools/cshogi_xref/hcpe3_to_psv.py \
    inbox/ref/sp_dr2-15K_20240210.hcpe3 \
    inbox/ref/sp_dr2-15K_20240210.hcpe3.psv 0 10000

# 全件 (max_games=0, max_positions=0)
python3 tools/cshogi_xref/hcpe3_to_psv.py \
    inbox/ref/full.hcpe3 \
    inbox/ref/full.hcpe3.psv
```

引数: `<hcpe3-in> <psv-out> [max_games] [max_positions]`

## Rust 側のクロスバリデーションテスト

### HCPE

`crates/bulletou_lib/src/value/loader/hcpe.rs::tests::cross_validate_against_cshogi_psv`。

```bash
python3 tools/cshogi_xref/hcpe_to_psv.py \
    inbox/ref/sp_dr2-15K_20240210.hcpe \
    inbox/ref/sp_dr2-15K_20240210.psv 10000

cargo test -p bulletou_lib --lib hcpe::tests::cross_validate_against_cshogi_psv -- --ignored --nocapture
```

期待される出力:

```
comparing 10000 records (hcpe total = 4583825, psv total = 10000)
OK: all 10000 records match byte-for-byte (BulletOu == cshogi)
```

### HCPE3

`crates/bulletou_lib/src/value/loader/hcpe3.rs::tests::cross_validate_against_cshogi_psv`。

```bash
python3 tools/cshogi_xref/hcpe3_to_psv.py \
    inbox/ref/sp_dr2-15K_20240210.hcpe3 \
    inbox/ref/sp_dr2-15K_20240210.hcpe3.psv 0 10000

cargo test -p bulletou_lib --lib hcpe3::tests::cross_validate_against_cshogi_psv -- --ignored --nocapture
```

期待される出力:

```
Rust side: decoded 10000 positions
cshogi side: psv file has 10000 records
OK: all 10000 HCPE3 records match byte-for-byte (BulletOu == cshogi)
```
