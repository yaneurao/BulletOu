# `bulletou_lib` を自分のコードから使う

<a href="../../en/advanced/bullet-lib.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

自分のコードに BulletOu の部品を組み込みたい人向けの補足です。初回の学習には不要です。

## 1. 既存 example を自分用に編集する

最も普通の使い方: リポジトリを clone して、[examples/](/examples) のいずれかを自分の目的に合わせて編集する。`shogi_simple.rs` や `bulletou.rs` を雛形に持つのが一般的。

## 2. 独自 example を登録する

新しい example ファイルを `examples/` 配下に置いただけでは `cargo build --example xxx` で認識されない。`bulletou_lib` の `Cargo.toml` に登録する必要がある:

```toml
# crates/bulletou_lib/Cargo.toml の末尾に追加
[[example]]
name = "my_example"
path = "../../examples/my_example.rs"
```

こうしておくと、上流からの `git pull` でファイルが消えにくく、独自実験を継続的に維持できる。

## 3. 他プロジェクトから `bulletou_lib` を import する

`bulletou_lib` を crate として他プロジェクトから依存することもできる:

```toml
[dependencies]
bullet = { git = "https://github.com/yaneurao/BulletOu", package = "bulletou_lib" }
```

## 4. API ドキュメント

詳細な API ドキュメントは Rust の docstring に書かれている。ローカルで生成・参照するには:

```bash
cargo doc --open
```

---

前へ: [応用編トップ](README.md)
