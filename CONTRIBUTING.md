# Contributing to BulletOu

Project conventions for code, documentation, and commits. This file
collects rules that we want consistently applied across the repository.

## Documentation style

### Markdown links

**Don't repeat the link target in the link's display text.** The path is
already encoded in the URL part of the link — duplicating it in the
visible label is noise.

Good:

```markdown
- [Tutorial](docs/en/tutorial/): walks through ...
- [Checkpoint layout spec](docs/spec/04-checkpoint-layout.md)
```

Bad:

```markdown
- [Tutorial (docs/en/tutorial/)](docs/en/tutorial/): walks through ...
- [Checkpoint layout (docs/spec/04-checkpoint-layout.md)](docs/spec/04-checkpoint-layout.md)
```

This also applies when the visible text is the bare URL: prefer
`<https://example.com/path>` (auto-link) over `[https://example.com/path](https://example.com/path)`.
