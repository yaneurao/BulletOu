# BulletOu の補助スクリプト

学習条件の探索や checkpoint の変換に使うスクリプトです。以下の例は **BulletOu のルートディレクトリ**から PowerShell で実行します。パスは手元の環境に合わせて変更してください。Python スクリプトの引数一覧は、それぞれ `--help` で確認できます。

## grid_search.py

指定したパラメーターの組み合わせを独立した trial として順番に学習し、結果を `grid_summary.csv` に集計します。共通設定には通常の BulletOu 設定 JSON を使います。

```powershell
python .\scripts\grid_search.py `
  --settings-file D:\path\bulletou-settings.json `
  --output-folder D:\path\grid-results `
  --max-epochs 1 --superbatches 32 `
  --lrs 0.0001 0.0002
```

`--dry-run` は学習せず実行計画を表示します。中断した実験は同じ出力先と条件を指定して `--resume` で再開できます。既存ログの集計だけなら `--output-folder D:\path\grid-results --summary-only` を指定します。詳細は [Grid search](../docs/ja/advanced/grid-search.md) を参照してください。

## tuning_parameters.py

短い試行を繰り返して学習率や factorizer などのパラメーターを探索し、generation ごとに選んだ条件で学習を進めます。設定にはチューニング用 JSON を使います。

```powershell
python .\scripts\tuning_parameters.py --settings-file D:\path\tuning-settings.json
```

`--dry-run` で設定を確認でき、`--resume` で中断した実験を再開できます。設定例と探索方法は [パラメーターチューニング](../docs/ja/advanced/parameter-tuning.md) を参照してください。

## bulletou_tuner.py

候補を世代ごとに生成し、各候補を固定長で学習して、最良の候補を次の世代へ引き継ぐチューナーです。`tuning_parameters.py` の共通処理としても使われます。

```powershell
python .\scripts\bulletou_tuner.py --tuning-settings-file D:\path\tuning-settings.json
```

`--dry-run` で実行計画を確認でき、`--resume` で再開できます。設定の内容と実装上の制約はスクリプト内および `--help` を参照してください。

## bulletou_csv.py

`summary-learn.csv` などのファイル名と、旧 `.log` 形式から `.csv` への名前変更を提供する共通モジュールです。`grid_search.py` や `bulletou_tuner.py` から読み込まれ、通常は直接実行しません。
