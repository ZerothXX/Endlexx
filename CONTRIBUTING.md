# Contributing / 参与开发

Read [README.en.md](README.en.md) or [README.md](README.md) first.
This source preview has no selected project-wide license; do not assume a contribution
will be distributed under MIT or another license. Clarify contribution terms with the owner
before submitting work intended for redistribution.

Use Python 3.12 and Node.js 22. From the source checkout:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check endlexx scripts tests
npm.cmd --prefix frontend run test:e2e
.\.venv\Scripts\python.exe scripts/package_source.py
```

Browser tests start an isolated API automatically unless `ENDLEXX_TEST_BASE_URL` is set.
If Python is not on PATH, set `ENDLEXX_TEST_PYTHON` to the absolute application interpreter.
Install Chromium first with `npm.cmd --prefix frontend exec -- playwright install chromium`.

Describe the problem, resulting behavior, relevant tests, and limitations in a pull request.
Tests should use isolated data and mocked providers unless explicitly testing a real model.
Keep model-quality failures visible; do not loosen assertions merely to obtain a passing result.
Do not commit `.env`, credentials, personal messages, images, model weights, or local runtimes.
The source archive includes neither legacy PPO training material nor private requirements notes.

请说明问题、修正后的行为、验证方法和保留限制。涉及真实模型的结果与边界单元测试应分开记录。

