# 09. cuda-oxide 高速化 TODO

tatara 同等速度を目標にした実装チケット。

この TODO は、既存 Bullet backend を壊さずに NNUE / SFNN 専用 cuda-oxide backend を
段階的に追加するための作業順である。各項目は小さい commit 単位に分割し、完了時に
status を更新する。

## TODO 一覧

| id | status | 内容 | 完了条件 |
|---|---|---|---|
| CO-001 | done | TODO 起票 | このファイルと README へのリンクを追加する |
| CO-002 | done | fixed-layout batch adapter | 既存 dataloader から `FastBatchHost` を直接列挙できる |
| CO-003 | in-progress | cuda-oxide crate 境界の作成 | 既存 workspace を巻き込まず、専用 crate / binary の置き場所を作る |
| CO-004 | todo | PTX smoke loader | 生成済み PTX を load し、最小 kernel を launch する |
| CO-005 | todo | CPU reference test harness | fast backend kernel と既存 Bullet backend の 1 batch 出力比較を作る |
| CO-006 | todo | minimal NNUE forward | `NNUE_HALFKP_256x2_32_32` の 1 batch forward を cuda-oxide で一致させる |
| CO-007 | todo | SFNN forward | `SFNN_halfka2_1024_7_64_k3k3` の forward を cuda-oxide で一致させる |
| CO-008 | todo | loss kernel | target transform / sigmoid / loss reduction を fused kernel 化する |
| CO-009 | todo | backward kernel | dense backward と sparse FT backward を実装する |
| CO-010 | todo | optimizer kernel | Ranger / RAdam update を fused kernel 化する |
| CO-011 | todo | async rings | input upload ring と loss readback ring を入れる |
| CO-012 | todo | checkpoint compatibility | `nn.bin` / log / checkpoint layout を既存と揃え、state backend marker を入れる |
| CO-013 | todo | speed benchmark | 同一 teacher / seed / schedule で existing Bullet backend と positions/sec を比較する |

## 作業原則

- 既存 `--backend bullet` は常に動く状態を保つ。
- cuda-oxide dependency は既存 workspace root に直接入れない。
- 数値が変わる高速化は opt-in にする。
- 速度比較は fp32 baseline の 1 batch 数値一致後に行う。
- KPPT / KPP_KKPT は今回の cuda-oxide 高速化対象外とする。
