# 1. クイックスタート — BulletOu をビルドして最小の学習を動かす

<a href="../../en/tutorial/1-quickstart.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

この章のゴール: 新規 clone した状態から BulletOu をビルドして、checkpoint ファイルを出力する小さな学習を動かす。ここまで通れば、ツールチェーン側は健全。

**強い NNUE を学習することが目的ではない** (それは次の章でやる)。これは smoke test。

## 1.1 必要なもの

以下が必要:

- **NVIDIA または AMD の GPU** (新しめのもの)。CPU だけでの学習はサポートされていない (GPU 前提の設計)
- **Rust ツールチェーン** (stable、1.87 以降)。OS 別の具体手順は §1.1.1 を参照
- **CUDA Toolkit 12.x** (NVIDIA GPU の場合) または **HIP SDK / ROCm** (AMD GPU の場合)
- ビルドとテストデータ用に **10 GB 程度の空きディスク**

Windows + NVIDIA の場合、cuDNN (および任意で TensorRT) のバージョンを揃える必要がある。詳細は本ワークスペース側の調査メモ ([../../../docs/spec/onnxruntime-gpu-windows.md](https://github.com/yaneurao/BulletOu)) を参照 (これは ONNX Runtime の話だが、CUDA 周りの DLL 設定は共通)。

> **CPU だけで動かしたい?** ソースツリーには `mock` GPU バックエンドがあるが、これは型チェック用のスタブで実際の学習はできない。GPU がない場合、このチュートリアルは動かない。クラウド GPU (Vast.ai / Lambda Labs / Paperspace / Google Colab 等) を借りるのが現実的。

### 1.1.1 Rust ツールチェーンのインストール

#### Windows

1. <https://rustup.rs/> から **rustup-init.exe** をダウンロード ("DOWNLOAD RUSTUP-INIT.EXE (64-BIT)" ボタン) して実行
2. デフォルト値で進める:
   - `Default host triple: x86_64-pc-windows-msvc` (CUDA EP との相性を考えて **msvc** ターゲットを推奨)
   - `Default toolchain: stable`
   - `Profile: default`
3. もし「MSVC C++ Build Tools が見つからない」とメッセージが出たら、誘導されるリンクから **Visual Studio Build Tools** をインストールし、「**C++ によるデスクトップ開発**」ワークロードにチェックを入れる
4. **PowerShell / cmd を一度開き直す** (PATH の更新を反映させるため。古いシェルでは `cargo` がまだ見えない)

PowerShell からの 1 行インストール (上記と等価):

```powershell
Invoke-WebRequest -Uri https://win.rustup.rs/x86_64 -OutFile rustup-init.exe
.\rustup-init.exe -y --default-host x86_64-pc-windows-msvc --default-toolchain stable
```

#### Linux / macOS / WSL

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"   # または新しいシェルを開く
```

#### 動作確認

```bash
cargo --version
rustc --version
```

両方とも `cargo 1.x.x ...` のようなバージョンが出れば OK。

## 1.2 ソースを取得する

```bash
git clone https://github.com/yaneurao/BulletOu.git
cd BulletOu
```

## 1.3 ビルド

GPU に応じて以下の **どちらか** を実行:

```bash
# NVIDIA GPU (CUDA)
CUDA_PATH=/usr/local/cuda cargo build --release --features device-cuda
```

```bash
# AMD GPU (ROCm)
HIP_PATH=/opt/rocm cargo build --release --features device-rocm
```

(Windows では `set CUDA_PATH=...` または PowerShell の `$env:CUDA_PATH=...` で環境変数を設定する。)

初回ビルドは数分〜十数分かかる。エラーなく完了すれば準備 OK。

### よくあるビルドエラー

- **`CUDA_PATH is not defined`** — 環境変数を CUDA のインストール先 (例: `/usr/local/cuda`、`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`) に設定する
- **`cublas` / `nvrtc` のリンクエラー** — CUDA のバージョンが古すぎる可能性。12.x 以降を使う
- **`hipblas` / `hiprtc` のリンクエラー** — HIP SDK をインストールし、`HIP_PATH` を設定する。場合により `GCN_ARCH_NAME` も設定が必要 (Linux なら `rocminfo`、Windows なら `hipinfo` で取得できる)

## 1.4 smoke test 用の学習を動かす

`simple` example は小さなチェス (将棋ではない) の NNUE を学習する。外部データが不要で、数分で終わる。パイプラインを端から端まで通すには十分。

```bash
# NVIDIA
cargo run --release --features device-cuda --example simple

# AMD
cargo run --release --features device-rocm --example simple
```

正常に動けば、以下のような出力が出る:

```
... starting training ...
superbatch 1 ... loss = ...
superbatch 2 ...
...
```

`checkpoints/` ディレクトリが作られ、学習結果のファイル群が書き出されている。

> `simple` example は **チェス** であって将棋ではない。上流由来で残っているもの。これがエンドツーエンドで動く最小の example なので、smoke test に使っている。将棋 example は次の章で。

## 1.5 今、何が起きたか

BulletOu をビルドして、完全な学習セッションを走らせた。動いたパイプライン:

1. `simple.rs` で小さな NNUE をビルド (チェス用 `Chess768` 入力特徴量 → 小さな隠れ層 → スカラー出力)
2. 同梱された小さなデータ (チェス用 `bulletformat`) を読み込み
3. 数 superbatch 分の学習
4. checkpoint を書き出し

次の章では、チェス用入力特徴量を **将棋用** に置き換え、実際の `.pack` データを使う。パイプラインは同じで、特徴量とデータローダーだけが違う。

## 1.6 後片付け

完了したら、`checkpoints/` と `target/` を削除して構わない:

```bash
rm -rf checkpoints target
```

`target/` は次回 `cargo build` のときに再生成される。

---

次へ: [2. 学習を走らせる](2-train.md) — 実データで評価関数を学習する。

---

<details>
<summary>1.7 開発者向け補足 (任意)</summary>

ここから先は smoke test を超えて、自分のコードに改造したい人向けの補足。**初回は読み飛ばして OK**。

### 既存 example を自分用に編集する

最も普通の使い方: リポジトリを clone して、[examples/](/examples) のいずれかを自分の目的に合わせて編集する。`shogi_simple.rs` や `bulletou.rs` を雛形に持つのが一般的。

### 独自 example を登録する

新しい example ファイルを `examples/` 配下に置いただけでは `cargo build --example xxx` で認識されない。`bullet_lib` の `Cargo.toml` に登録する必要がある:

```toml
# crates/bullet_lib/Cargo.toml の末尾に追加
[[example]]
name = "my_example"
path = "../../examples/my_example.rs"
```

こうしておくと、上流からの `git pull` でファイルが消えにくく、独自実験を継続的に維持できる。

### 他プロジェクトから `bullet_lib` を import する

`bullet_lib` を crate として他プロジェクトから依存することもできる:

```toml
[dependencies]
bullet = { git = "https://github.com/yaneurao/BulletOu", package = "bullet_lib" }
```

### API ドキュメント

詳細な API ドキュメントは Rust の docstring に書かれている。ローカルで生成・参照するには:

```bash
cargo doc --open
```

</details>
