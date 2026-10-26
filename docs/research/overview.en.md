# Research and application overview

[中文](overview.md) · [English](overview.en.md) · [Project home](../../README.en.md) · [Results and limitations](results.en.md)

Endlexx is the main project entry and local agent workbench. [Endless-XXX](https://github.com/ZerothXX/Endless-XXX) is the separate character visual-transfer research repository. This section presents the method, integrated capabilities and version boundaries without maintaining a second training core.

## Research question and method

The aim is to transfer character hair, clothing, colors and recognizable accessories while preserving the input subject class, visible regions and composition. People can adopt character appearance; animals and objects should express the association through colors, patterns and accessories rather than all becoming humans.

1. Reference images and per-image captions supervise SDXL UNet LoRA training. The base model, VAE and text encoders remain frozen.
2. Reviewed character cards define identity; the VLM analyzes input subjects, visible regions and materials to reduce identity contamination from recognition errors.
3. img2img anchors the input, ControlNet-Canny supplies structural conditions, and optional SAM2 masks support segmentation. Different subject types use different strengths and prompt constraints.
4. Outputs, prompts and subject/planning/generation records support comparisons with fixed inputs, seeds and weights. Diagnostic training loss is not a generation-quality metric.

Research code, model layout, standard training and the 24GB configuration entry are documented in the [research repository](https://github.com/ZerothXX/Endless-XXX#readme). Application retrieval, restoration and context engineering are described on the [Endlexx homepage](../../README.en.md).

## Repository responsibilities

| Content | Maintained in | Relationship to the application |
|---|---|---|
| Training loop, preprocessing, research inference policy | Endless-XXX | Snapshot updates require explicit version evaluation |
| Agent, RAG, chats, context and assets | Endlexx | Runs independently; image execution requires separate models |
| Inference adapter, GPU scheduling, task logs | Endlexx | Calls an installed research entry or the bundled snapshot |
| Historical research gallery | Originals in Endless-XXX; selected copies here | Traced to a commit and hashes; not application acceptance evidence |
| LoRA, character packages, third-party models | Prepared/published separately | Not distributed in the source archive |

## Distinct versions

| Item | Source pinned by this documentation | Meaning |
|---|---|---|
| Historical images and research README baseline | [Endless-XXX `6bd78d48c4c99bef0be3ecf4f71e8536349c2acb`](https://github.com/ZerothXX/Endless-XXX/tree/6bd78d48c4c99bef0be3ecf4f71e8536349c2acb) | Online revision read on 2026-10-26; no new inference run |
| Bundled application inference | [vendor/endless](../../vendor/endless/README.md), local snapshot dated 2026-10-24 | Identified by its [file hashes](../../vendor/endless/source-manifest.json), not mapped to the public commit above |
| Application model evaluation | [2026-10-25 development regression](../Agent质量与改进验收.md) | Agent planning and related tasks, separate from the gallery |

The public research revision uses `character.json` package metadata. The application snapshot comes from another local revision; matching directory names do not establish interchangeability. `ENDLEXX_ENDLESS_ROOT` can select an existing research installation, but does not guarantee compatibility with arbitrary commits. Check trigger words, base models, LoRA format and package conventions before using the published March 7th weights.

## Criteria for future integration

- Record the upstream commit or local file hashes when updating the snapshot; keep one upstream training implementation.
- Specify package schemas, weight selection, trigger words and base-model dependencies. Copying a weight file alone does not prove compatibility.
- Validate installation and fallback in isolation, then cover people, animals, objects, visible-region variations and failure recovery.
- Keep actual seeds/settings, model/weight hashes and input provenance; manually review structure, identity, accessories and background.
- Mark in-app training as implemented only after training jobs, status, cancellation, outputs and recovery are integrated and tested.

These are future acceptance criteria. The homepage redesign does not implement missing capabilities. Current runtime and quality limits remain in [release verification](../RELEASE.md) and the [results page](results.en.md).
