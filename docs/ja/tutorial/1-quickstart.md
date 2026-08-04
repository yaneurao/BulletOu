# 1. クイックスタート — BulletOu をビルドして最小の学習を動かす

<a href="../../en/tutorial/1-quickstart.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

この章のゴール: 新規 clone した状態から BulletOu をビルドして、checkpoint ファイルを出力する小さな学習を動かす。ここまで通れば、ツールチェーン側は健全。

**強い NNUE を学習することが目的ではない** (それは次の章でやる)。これは smoke test。

## 1.1 必要なもの

以下が必要:

- **NVIDIA GPU** (CUDA 対応のもの)。CPU だけでの学習はサポートされていない (保守対象 backend は CUDA 前提)
- **Rust ツールチェーン** (stable、1.87 以降)。OS 別の具体手順は §1.1.1 を参照
- **CUDA Toolkit 12.x**
- ビルドとテストデータ用に **10 GB 程度の空きディスク**

Windows の場合、Cargo を実行する shell から MSVC C++ build tools も見える必要がある。

> **CPU だけで動かしたい?** ソースツリーには `mock` GPU バックエンドがあるが、これは型チェック用のスタブで実際の学習はできない。GPU がない場合、このチュートリアルは動かない。クラウド GPU (Vast.ai / Lambda Labs / Paperspace / Google Colab 等) を借りるのが現実的。

### 1.1.1 Rust ツールチェーンのインストール

#### Windows

1. <https://rustup.rs/> から **rustup-init.exe** をダウンロード ("DOWNLOAD RUSTUP-INIT.EXE (64-BIT)" ボタン) して実行
2. デフォルト値で進める:
   - `Default host triple: x86_64-pc-windows-msvc` (CUDA EP との相性を考えて **msvc** ターゲットを推奨)
   - `Default toolchain: stable`
   - `Profile: default`
3. もし「MSVC C++ Build Tools が見つからない」とメッセージが出たら、誘導されるリンクから **Visual Studio Build Tools** をインストールし、「**C++ によるデスクトップ開発**」ワークロードにチェックを入れる
4. **PowerShell / cmd を一度開き直す** (PATH の更新を反映させるため。開き直す前のシェルでは `cargo` がまだ見えない)

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

BulletOu の学習 backend をビルドする:

```bash
# NVIDIA GPU (CUDA 12.x)
cargo build --release --features cuda-cpp-backend --example bulletou
```

Windows では CUDA Toolkit と対応する Visual Studio C++ build tools がビルド環境から見えるようにしておく。

初回ビルドは数分〜十数分かかる。エラーなく完了すれば準備 OK。

### よくあるビルドエラー

- **`CUDA_PATH is not defined`** — 環境変数を CUDA のインストール先 (例: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`、`/usr/local/cuda`) に設定する
- **Windows で `nvcc` / MSVC build error が出る** — CUDA Toolkit と Visual Studio C++ build tools を入れ、両方が見える shell から実行する
- **CUDA runtime library のリンクエラー** — CUDA のバージョンが古すぎる可能性。12.x 以降を使う

## 1.4 smoke test 用の学習を動かす

CUDA C++ smoke test は教師データ不要。CUDA 初期化、kernel launch、小さな Ranger update が通るかを確認する。

```bash
cargo run --release --features cuda-cpp-backend --example bulletou -- --cuda-cpp-smoke
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

次へ:
- [2. `bulletou_lib` を自分のコードから使う](2-bullet-lib.md) — 開発者向けの補足 (任意)
- 学習を進めたい場合は [3. 教師データを用意する](3-data.md) へ
