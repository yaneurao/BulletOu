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

このツールは CUDA や HIP を **必要としない**。

### バックエンド

#### CUDA

NVIDIA GPU を持っているユーザー向け。

- `cuda` feature を有効化する
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) をインストールする
  - できるだけ新しいバージョンを推奨
  - Toolkit のバージョンが古すぎる場合、コンパイル時にリンカエラーが出るか、実行時に比較的わかりやすいエラーで知らされる
- 環境変数 `CUDA_PATH` を CUDA のインストール先 (`bin`, `lib`, `include` ディレクトリを含むはずのパス) に設定する必要がある

#### ROCm

AMD GPU を持っているユーザー向け。

- `rocm` feature を有効化する
- [HIP SDK](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html) をインストールする
- 環境変数 `HIP_PATH` を HIP のインストール先 (`bin`, `lib`, `include` ディレクトリを含むはずのパス) に設定する必要がある
- 環境変数 `GCN_ARCH_NAME` の設定も必要な場合が多い。Linux なら `rocminfo`、Windows なら `hipinfo` で確認できる
