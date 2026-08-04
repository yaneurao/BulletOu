# 1. クイックスタート

<a href="../../en/tutorial/1-quickstart.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

ゴール: BulletOu をビルドして、CUDA backend が動くことを確認します。

## 1.1 必要なもの

- NVIDIA GPU
- CUDA Toolkit 12.x
- Rust stable
- Windows では Visual Studio C++ Build Tools

CPU だけでの学習はサポートしていません。

Rust は <https://rustup.rs/> から入れられます。インストール後、新しい PowerShell を開いて次を確認してください。

```powershell
cargo --version
rustc --version
```

## 1.2 ソースを取得する

```powershell
git clone https://github.com/yaneurao/BulletOu.git
cd BulletOu
```

すでに clone 済みなら、この手順は不要です。

## 1.3 ビルドする

```powershell
cargo build --release --features cuda-cpp-backend --example bulletou
```

Windows では、実行ファイルは次にできます。

```text
.\target\release\examples\bulletou.exe
```

## 1.4 CUDA smoke test

教師データなしで、CUDA 初期化と小さな学習 kernel を確認できます。

```powershell
cargo run --release --features cuda-cpp-backend --example bulletou -- --cuda-cpp-smoke
```

エラーなく終了すれば、ビルド環境はひとまず正常です。

## 1.5 よくあるビルドエラー

| エラー | 確認すること |
| --- | --- |
| `CUDA_PATH is not defined` | CUDA Toolkit のインストール先が環境変数に入っているか |
| `nvcc` が見つからない | CUDA Toolkit を入れた PowerShell を開き直したか |
| MSVC 関連のエラー | Visual Studio C++ Build Tools が入っているか |

---

次へ: [2. 教師データを用意する](2-data.md)
