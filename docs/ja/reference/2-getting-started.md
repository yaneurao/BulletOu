# はじめかた

<a href="../../en/reference/2-getting-started.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

### Rust をインストールする

[rustup](https://www.rust-lang.org/tools/install) を使って Rust をインストールする (これが公式の Rust インストール方法)。

### 一般的な使い方

`bullet` (BulletOu 内の crate 名は本家との衝突を避けるため `bulletou_lib` にリネーム済み) を crate として使うことができる:

```toml
bullet = { git = "https://github.com/yaneurao/BulletOu", package = "bulletou_lib" }
```

または、[examples](../../examples) のいずれかを編集して実行する:

```
cargo r -r --example <example name>
```

最低限の例として [examples/simple](../../examples/simple.rs) が同梱されている。NNUE の学習を初めて行うなら、これに近いアーキテクチャと学習スケジュールから始めることを推奨。

### ユーティリティ

`bullet-utils` を以下のコマンドでビルドできる:

```
cargo b -r --package bullet-utils
```

`bullet-utils` でできること:

- データフォーマット間の変換
- 複数のデータファイルの interleave (交互混合)
- データファイルのシャッフル
- データファイルの検証 (validate)

具体的な使い方は以下で確認:

```
./target/release/bullet-utils[.exe] help
```

このツールは CUDA を **必要としない**。

### バックエンド

現在保守している BulletOu の学習 backend は NVIDIA GPU 用の `cuda-cpp`:

```bash
cargo build --release --features cuda-cpp-backend --example bulletou
```

- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) をインストールする
- 環境変数 `CUDA_PATH` を CUDA のインストール先に設定する
- ROCm/HIP 対応と旧 `bullet-gpu` feature backend は、現在の保守ビルドから削除済み
