# BulletOu Specifications

このディレクトリは BulletOu (やねうら王対応の Rust ML トレーナー) の **仕様レベルのリファレンス** を集約する場所です。

「やね評価関数を学習する側」「やねうら王エンジンで学習結果を load する側」「BulletOu の coding をする側」の **3 者の契約 (contract)** を明文化することを目的とする。

## docs/ 配下の他ディレクトリとの違い

| | docs/tutorial, docs/shogi 他 | docs/spec/ (このディレクトリ) |
|---|---|---|
| 対象 | エンドユーザー (学習を回す人) | 実装者・engine 連携者・後続コミッター |
| 内容 | チュートリアル / コマンド例 / ベストプラクティス | バイナリ仕様 / hash 値 / dim 表 / 規約 |
| 性質 | 「使い方」 | 「仕様」 |
| 改訂頻度 | 機能追加で更新 | 仕様変更 (= breaking change) でのみ更新 |

ここに書く情報は **コードを読まなくても再実装できる粒度** であること。具体的な const 値 (FEATURE_HASH 等) や bit layout は必ず確定値で記述する。

## ドキュメント一覧

- [01-eval-types.md](01-eval-types.md) — `--eval-type` の公開バリアントと出力ファイル組合せ
- [02-nnue-binary.md](02-nnue-binary.md) — `nn.bin` (NNUE binary) のヘッダー / 重みレイアウト / 量子化スケール / hash 計算式
- [03-feature-sets.md](03-feature-sets.md) — HalfKP / K / P / FeatureSet 合成規則と発火 index 配置
- [04-checkpoint-layout.md](04-checkpoint-layout.md) — 番号付き save dir / `state.bin` / `learn.log` / resume プロトコル
- [05-activation-history.md](05-activation-history.md) — ClippedReLU と SqrClippedReLU の歴史的経緯 (誤実装回避用のメモ)
- [06-validation-metrics.md](06-validation-metrics.md) — `test_value_accuracy` / `test_value_loss` の定義と YaneuraOu / dlshogi との cross-tool 数値比較契約
