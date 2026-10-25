import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from endlexx.providers import runtime_call

print(
    json.dumps(
        runtime_call(
            "/generate",
            {"messages": [{"role": "user", "content": '只输出JSON：{"ok":true}'}], "max_tokens": 40},
        ),
        ensure_ascii=False,
    )
)
