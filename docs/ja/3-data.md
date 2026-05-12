# 教師データ

<a href="../en/3-data.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

## 標準的なワークフロー

1. 何らかのフォーマットでデータを保存する (binpack 系のフォーマットを強く推奨)
2. 必要に応じて、保存したフォーマットから `Trainer` が読み込めるデータフォーマットへ変換する
3. 変換した個々のファイルをシャッフルする
4. シャッフル済みのファイルを interleave (交互混合) する

## 同梱のデータローダー

独自フォーマット用のデータローダーは簡単に書けるが、bullet には主要なフォーマット用のローダーが既に同梱されている。

### binpack

Stockfish / Monty / Viridithas の binpack は、それぞれ `SfBinpackLoader` / `MontyBinpackLoader` / `ViriBinpackLoader` で読み込める。binpack はゲーム単位でデータを連続して格納することで高い圧縮率を実現しているため、ノイズの多い局面などを除外するフィルタ関数を渡す必要がある。

自前でデータを生成するユーザーの間で最も広く使われている **Viriformat binpack** を推奨する (多くのプログラミング言語に参照実装があり、関連ユーティリティも豊富)。

### ChessBoard (別名 "bulletformat")

`DirectSequentialDataLoader` で読み込める、シンプルで高速にロードできるデータフォーマット。小さなネットワークの学習に向く。

ただし、データの生成・保存は binpack 系で行い、データロード速度がボトルネックになって初めてこのフォーマットに変換することを推奨。このフォーマットは手番情報 (record は stm 視点で保存される)、halfmove カウンタ、キャスリング権などの情報を捨てている。

`bullet-utils` バイナリには、このフォーマットのファイルをシャッフル・interleave するユーティリティと、他のいくつかのフォーマットからの変換ツールが含まれる。

特に、以下の形式のテキストファイル (1 局面 1 行) から変換できる:

- 各行は `<FEN> | <score> | <result>` の形式
- `score` は白視点の centipawn 単位
- `result` は白視点で、勝ち `1.0` / 引き分け `0.5` / 負け `0.0` の形式
