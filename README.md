# comp_model

Start with [example/README.md](example/README.md).

- [example/models/README.md](example/models/README.md): fixed per-model workflow scripts
- [example/model_recovery/](example/model_recovery/): model comparison and recovery studies

## Node.js for type checking

Pyright uses Node.js. Before running `uv run pyright src/`, select the patched
Node.js 24.22.1 LTS runtime from the repository root with
[nvm](https://github.com/nvm-sh/nvm):

```bash
nvm install
nvm use
```

CI installs the same version from `.nvmrc` before the type check. The Python
modeling library itself does not require Node.js.
