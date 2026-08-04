# Using `bulletou_lib` from your own code

<a href="../../ja/advanced/bullet-lib.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

This page is for using BulletOu components from your own code. It is not needed for a first training run.

## 1. Editing an existing example

The usual workflow: clone the repo and edit one of the files under [examples/](/examples) to your taste. `shogi_simple.rs` or `bulletou.rs` are common starting points.

## 2. Registering a custom example

Just placing a new file under `examples/` is not enough — `cargo build --example xxx` will not find it until you register it in `bulletou_lib`'s `Cargo.toml`:

```toml
# Append to crates/bulletou_lib/Cargo.toml
[[example]]
name = "my_example"
path = "../../examples/my_example.rs"
```

Once registered, the example survives `git pull` from upstream more easily, which helps when maintaining long-running custom experiments.

## 3. Importing `bulletou_lib` from another project

You can also depend on `bulletou_lib` as a crate from a separate project:

```toml
[dependencies]
bullet = { git = "https://github.com/yaneurao/BulletOu", package = "bulletou_lib" }
```

## 4. API documentation

Detailed API documentation lives in Rust's docstrings. To generate and open it locally:

```bash
cargo doc --open
```

---

Previous: [Advanced guide](README.md)
