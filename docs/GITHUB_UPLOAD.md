# GitHub 上传说明 / upload guide

## Endlexx：完整源码 / full source

解压 `Endlexx-0.1.0-source.zip`，将 `Endlexx-0.1.0/` **里面的文件**提交到 [Endlexx 仓库](https://github.com/ZerothXX/Endlexx) 根目录。保留 `.github/`、`.gitignore`、`.gitattributes`、`.env.example` 等隐藏项；不要把整个外层目录再嵌套一层。

Extract and commit the **contents** of `Endlexx-0.1.0/` to the repository root, including hidden files. The archive is full publishable application source plus documentation and registered gallery media. It excludes weights, runtimes and private user data. A ZIP uploaded as a Release attachment does not itself update the repository homepage.

提交后检查 / After committing:

- 中文和英文 README 能互相切换，三组图片和研究文档可打开。
- GitHub Actions 在目标提交运行；本地测试通过不代表远程 CI 已运行。
- README 的研究原型、兼容性与效果限制仍清楚显示。

## Endless-XXX：文档覆盖包 / documentation overlay

另一个 `Endless-XXX-readme-update-2026-10-26.zip` 包仅含文档更新和核对说明。只将其中 `Endless-XXX/README.md`、`Endless-XXX/README.en.md` 提交到 [研究仓库](https://github.com/ZerothXX/Endless-XXX) 根目录，保留原有研究源码和展示图。它不是完整研究仓库源码包，不应用来清空仓库。

The separate overlay replaces/adds the two root README files only. Preserve research code and images. Read its `UPDATE_INSTRUCTIONS.md` and compare `UPDATE_MANIFEST.json` before applying. It targets public baseline `6bd78d48c4c99bef0be3ecf4f71e8536349c2acb`, not every local research revision. Merge newer changes first. Commit Endlexx first so the new cross-repository documentation targets exist.

## GitHub 页面设置 / optional repository settings

这些是页面设置建议，不属于源码文件，也不会由打包脚本自动应用。
These are suggestions for GitHub's repository settings; the archive does not apply them.

| 仓库 / Repository | About description suggestion | Topics suggestion |
|---|---|---|
| Endlexx | Character creation agent workbench and main hub for Endless-XXX visual transfer research. | `agent`, `character-creation`, `rag`, `visual-transfer`, `sdxl` |
| Endless-XXX | Research core for character LoRA training and visual transfer across open-domain subjects. Main application: Endlexx. | `lora`, `sdxl`, `controlnet`, `image-to-image`, `visual-transfer` |

研究仓库 About 的 Website 可填主项目地址；无需迁移仓库、合并 Git 历史或启用 Pages 才能呈现这一关系。README 素材已使用本仓库内路径，来源链接固定到研究提交，避免依赖外部图床。

The research repository's About website can point to Endlexx. Repository migration, history merging and Pages are not required for this presentation. Endlexx gallery images use local paths with pinned upstream provenance.

## 发布前本地检查 / local checks

```powershell
.\.venv\Scripts\python.exe scripts/check_publication.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check endlexx scripts tests
.\.venv\Scripts\python.exe scripts/package_source.py
.\.venv\Scripts\python.exe scripts/package_source.py --verify dist/Endlexx-0.1.0-source.zip
```

源码清单与 ZIP SHA256 可检测内容变化，不是数字签名或独立审计。许可证状态仍见 [LICENSE.md](../LICENSE.md)，素材来源与未记录的权限见 [第三方说明](../THIRD_PARTY_NOTICES.md)。
