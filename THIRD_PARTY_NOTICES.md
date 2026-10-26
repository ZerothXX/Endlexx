# Sources and third-party notices / 来源与第三方说明

This is a provenance inventory, not a blanket license grant.
这是来源清单，不是对所有组件的统一授权。

| Component | Provenance and distribution |
|---|---|
| Endlexx application | Application code in `endlexx/`, `frontend/`, and `scripts/`. Project license is not yet specified; see `LICENSE.md`. |
| Endless inference snapshot | `vendor/endless/` contains the project owner's local Endless inference snapshot captured on 2026-10-24. Per-file hashes are preserved in `source-manifest.json`. No standalone license was supplied with this snapshot; this release does not assert a third-party license or relicense it. |
| PPO research | Separate [PPO restoration research](https://github.com/ZerothXX/Detail-Enhancement-for-Anime-Image-Restoration-Based-on-PPO-Post-processing). Legacy training code/data/checkpoints are excluded from this source archive. PPO is not enabled in the application. |
| Qwen / Ollama / PyTorch / Diffusers / LaMa / BGE | Downloaded separately at runtime or installed as dependencies. No model weights or executables are bundled. Their upstream licenses and model cards apply independently. |
| Published research gallery | Selected byte-for-byte copies from [Endless-XXX](https://github.com/ZerothXX/Endless-XXX), commit `6bd78d48c4c99bef0be3ecf4f71e8536349c2acb`, under `docs/media/research/`. See the [provenance manifest](docs/media/research/manifest.json). Upstream describes input images as sourced from the internet; original authors and individual image licenses were not recorded. No new image license is asserted. These historical illustrations are not application acceptance results. |
| Character packages, user reference images, datasets | Not included. Users supply assets and role-specific weights separately and are responsible for their rights. |

Application dependencies are declared in `pyproject.toml`, `requirements.lock`,
`requirements-model.txt`, `requirements-bundle.txt`, and `frontend/package-lock.json`.
The dependency license notices remain in their installed distributions.
Downloading a model is not evidence that its license permits every intended use.

源码包不含模型、角色 LoRA、私人用户图片、数据集、数据库或第三方运行时；仅附有来源清单中指定的公开研究展示图，素材权利不因收录而改变。
研究快照仅保留已纳入项目的推理源码；不对其作 MIT 等未经指定的授权声明。

