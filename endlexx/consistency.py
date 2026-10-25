"""Derive verdicts from explicit visual observations, preserving their evidence."""

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field


class Observation(BaseModel):
    feature: str = Field(min_length=1, max_length=160)
    expected: str = Field(max_length=2000)
    support_region: str = Field(min_length=1, max_length=200)
    region_visibility: Literal["clear", "occluded", "outside", "uncertain"]
    comparison: Literal["match", "mismatch", "uncertain"]
    observation: str = Field(min_length=1, max_length=2000)


def consistency_report(raw: dict) -> dict:
    if not isinstance(raw.get("items"), list) or not 1 <= len(raw["items"]) <= 40:
        raise ValueError("一致性检查返回的观察条目不正确")
    items = []
    for value in raw["items"]:
        item = Observation.model_validate(value)
        if item.region_visibility in {"occluded", "outside"}:
            status = "not_visible"
        elif item.region_visibility == "clear" and item.comparison != "uncertain":
            status = item.comparison
        else:
            status = "unknown"
        items.append(item.model_dump() | {"status": status})
    counts = Counter(item["status"] for item in items)
    summary = "；".join(
        f"{label} {counts[key]} 项"
        for key, label in [
            ("match", "可见且符合"),
            ("mismatch", "可见但不符"),
            ("not_visible", "部位不可见"),
            ("unknown", "无法判断"),
        ]
    )
    return {"items": items, "summary": summary + "。判断依据来自视觉模型，请结合原图核对。"}


def review_candidates(report: dict) -> list[dict]:
    # Re-observe the image from the requirements alone. Showing the first
    # uncertain verdict/observation makes small models repeat it verbatim.
    return [
        {"index": index, **{key: item[key] for key in ("feature", "expected", "support_region")}}
        for index, item in enumerate(report["items"])
        if item["status"] == "unknown"
    ][:4]


def merge_review(initial: dict, raw: dict) -> dict:
    items = [dict(item) for item in initial["items"]]
    candidates = raw.get("items", [raw] if "feature" in raw else [])
    accepted = []
    if isinstance(candidates, list):
        for value in candidates[:4]:
            if not isinstance(value, dict):
                continue
            index = value.get("index")
            if (
                type(index) is not int
                or not 0 <= index < len(items)
                or initial["items"][index]["status"] != "unknown"
                or index in accepted
            ):
                continue
            try:
                observed = Observation.model_validate(value).model_dump()
            except ValueError:
                continue
            items[index] = observed | {
                "feature": initial["items"][index]["feature"],
                "expected": initial["items"][index]["expected"],
            }
            accepted.append(index)
    return consistency_report({"items": items}) | {
        "initial_items": initial["items"],
        "reviewed": bool(accepted),
        "review": {
            "raw": raw,
            "accepted_indices": accepted,
            "warning": None if accepted else "复核未提供有效补充观察，保留初次判断。",
        },
    }
