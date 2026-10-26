# Release verification / 发布验收

Date / 日期：2026-10-26。Version / 版本：0.1.0 research preview。

## Release scope / 发布范围

Suitable for a GitHub source preview of a single-user local research workbench.
This is not a stable production release, a hosted service or a complete offline installer.
The UI is Chinese; both Chinese and English README files describe setup and limitations.

可作为单人本机研究项目在 GitHub 展示、下载和继续开发。迁移画质与部分一致性判断
尚未通过效果验收，不将其标为稳定功能。不含个人资料、模型、运行时或角色 LoRA。

## Changes / 发布补齐

- Rewrote Chinese README and added the English counterpart with matching claims.
- Documented empty-install behavior; development-machine model connections are not shipped.
- Added contribution, security, provenance, license-status and changelog documents.
- Expanded GitHub Actions to Windows/Linux application checks and an isolated browser job.
- Browser tests start their own API/database and close it automatically.
- Windows process tests are explicitly platform-scoped, avoiding accidental Unix signals.
- Added allowlisted source packaging with high-confidence credential-pattern checks,
  per-file SHA256 verification, deterministic ZIP metadata and an archive checksum.
- Excluded local databases, private user assets, keys, runtimes, caches, unused PPO training material,
  private requirement notes and architecture-editor metadata.
- Kept the vendor inference snapshot and its original hash inventory.
- Included synthetic raw model-evaluation reports under `evals/reports/`.

## Original runtime verification / 原运行时验证

| Check / 检查 | Local result / 本机结果 |
|---|---|
| Python regression tests | 86 passed, 12.47 seconds; includes release privacy/reproducibility/tamper checks |
| Ruff | Passed for `endlexx scripts tests` |
| Frontend | TypeScript + Vite build passed, both working tree and extracted source |
| Browser tests | 9 passed, 14.4 seconds; managed isolated API stopped after completion |
| Python package metadata/build | Wheel built from extracted source with an isolated setuptools build environment |
| Source archive | Allowlisted contents, per-file hashes, original vendor hashes, Markdown links, credential patterns and deterministic ZIP verification |

Initial browser launch was blocked by the execution sandbox (`spawn EPERM`); the same
tests passed when run outside that sandbox. An initial `--no-build-isolation` wheel
attempt lacked setuptools in the existing application environment; the standard
isolated build installed that build dependency and succeeded. Neither failure was
hidden or treated as a passing test.

The remote GitHub workflow has **not run** because this task produces a local source
package. Linux CI is configured, not locally certified on this Windows machine.
Existing application dependencies were reused for the extracted-source startup
check; this does not repeat a blank Windows installation or full model download.
No new model-quality evaluation is claimed beyond the recorded 2026-10-25 experiments.

## Research presentation update / 研究展示更新

Endlexx now presents both the application and Endless-XXX research on its Chinese/English
homepages. Bilingual research overview/results pages distinguish research training from
application integration and document version compatibility, failures and reproduction gaps.
Seven historical public images were copied unchanged from research commit
`6bd78d48c4c99bef0be3ecf4f71e8536349c2acb`. Their individual rights and complete historical
generation settings are not recorded; see the [provenance inventory](media/research/manifest.json).

源码包新增研究说明和指定展示素材，打包时检查本地文档链接、素材清单、SHA256、
Git blob 标识和文件大小；未经登记的用户图片不随之纳入。另提供对应线上研究基线的
Endless-XXX 双语 README 覆盖包，保留其原有代码和图片，不覆盖不同修订的本机研究目录。

The new documentation/packaging checks are separate from the original runtime evidence above.
The updated application suite passed **88 tests in 20.31 seconds**; Ruff passed for
`endlexx scripts tests`. The focused source-release suite passed **10 tests** including
broken publication targets, registered-media inclusion, private-media exclusion and altered
image rejection. The two research README files have **36 relative file references** checked
against the pinned public Git tree plus the added English file. These counts describe local
checks, not remote GitHub execution or a fresh model-quality experiment.
The application publication pass checks **9 documents, 98 relative file references and
7 approved research images**, including checks that all referenced local targets are actually
present in the source archive. Original runtime test counts above remain historical evidence.

No frontend/runtime algorithm was changed, and model inference/training and browser tests
were not rerun for this presentation update. GitHub About/topic suggestions and exact upload
destinations are documented in the [upload guide](GITHUB_UPLOAD.md); no remote settings or
repository contents were changed by this local task.

## Model quality / 模型效果

The 2026-10-25 development set is **not held out**. Qwen3 8B achieved 15/16 tool selection,
7/8 complex planning and 6/6 summary fidelity with application safeguards; the original
Qwen3-VL 4B achieved 16/16, 5/8 and 6/6. Raw reports:
[native](../evals/reports/native.json), [text 8B](../evals/reports/text8b.json).
See [detailed quality report](Agent质量与改进验收.md).

Transfer still produces visible banding/artifacts. Missing glasses can remain `unknown`
in consistency checks. PPO is not enabled. External APIs were tested at the protocol
boundary without a real provider key. These limitations remain visible in both README files.

## License and upload / 许可证与上传

No project-wide license has been selected. This preview does not assert an MIT/Apache
license or relicense research code; see [license status](../LICENSE.md) and
[source inventory](../THIRD_PARTY_NOTICES.md). GitHub distinguishes publishing source from
licensed open source in its [licensing documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

Run `python scripts/package_source.py`, then verify the ZIP with `--verify`.
Extract it and commit the contents of `Endlexx-0.1.0/` into the GitHub repository root,
including hidden files. Do not upload the working directory wholesale or commit the
newly generated `.env`/`data/` after running the application.

The checksum and file manifest detect content changes; they are not a digital signature
or an independent security audit. No remote repository, commit, tag or Release is created
by the packaging script.
