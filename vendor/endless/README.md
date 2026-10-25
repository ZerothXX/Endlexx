# Endless inference snapshot

This directory contains an application-owned snapshot of the user's sibling
`Endless_xxx` inference source, captured on 2026-10-24. SHA256 values are recorded
in `source-manifest.json`. It includes the formal `test.py` entry and its local
dependencies, without model weights, datasets, role packages, training entry,
web application, credentials or repository instructions.

The installer copies this snapshot into `data/bundle/endless` only when no complete
research installation is available. It never installs weights into the sibling
research repository. Role-specific LoRA weights must be published/imported
separately. Updating the research algorithms requires an explicit new snapshot;
this directory is not a separately maintained fork of the research project.
