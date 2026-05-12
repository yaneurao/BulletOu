# 2. Using `bullet_lib` from your own code (optional)

<a href="../../ja/tutorial/2-bullet-lib.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Material beyond the smoke test, for when you start adapting BulletOu to your own training. **Safe to skip on first read** — feel free to jump to [3. Prepare training data](3-data.md).

## 2.1 Editing an existing example

The usual workflow: clone the repo and edit one of the files under [examples/](/examples) to your taste. `shogi_simple.rs` or `bulletou.rs` are common starting points.

## 2.2 Registering a custom example

Just placing a new file under `examples/` is not enough — `cargo build --example xxx` will not find it until you register it in `bullet_lib`'s `Cargo.toml`:

```toml
# Append to crates/bullet_lib/Cargo.toml
[[example]]
name = "my_example"
path = "../../examples/my_example.rs"
```

Once registered, the example survives `git pull` from upstream more easily, which helps when maintaining long-running custom experiments.

## 2.3 Importing `bullet_lib` from another project

You can also depend on `bullet_lib` as a crate from a separate project:

```toml
[dependencies]
bullet = { git = "https://github.com/yaneurao/BulletOu", package = "bullet_lib" }
```

## 2.4 API documentation

Detailed API documentation lives in Rust's docstrings. To generate and open it locally:

```bash
cargo doc --open
```

---

Next: [3. Prepare training data](3-data.md)

Previous: [1. Quick Start](1-quickstart.md)
