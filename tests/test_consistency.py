from endlexx.consistency import consistency_report, merge_review, review_candidates


def test_consistency_distinguishes_missing_accessory_from_cropped_support_region():
    common = {"feature": "饰品", "expected": "红色耳饰", "support_region": "耳朵", "observation": "像素观察"}
    report = consistency_report(
        {
            "items": [
                common | {"region_visibility": "clear", "comparison": "mismatch"},
                common | {"region_visibility": "outside", "comparison": "mismatch"},
                common | {"region_visibility": "occluded", "comparison": "match"},
                common | {"region_visibility": "uncertain", "comparison": "match"},
                common | {"region_visibility": "clear", "comparison": "uncertain"},
                common | {"region_visibility": "clear", "comparison": "match"},
            ]
        }
    )
    assert [item["status"] for item in report["items"]] == [
        "mismatch",
        "not_visible",
        "not_visible",
        "unknown",
        "unknown",
        "match",
    ]


def test_optional_review_preserves_valid_report_on_empty_or_malformed_response():
    item = {
        "feature": "配饰",
        "expected": "红色耳饰",
        "support_region": "耳朵",
        "observation": "尚不确定",
        "region_visibility": "uncertain",
        "comparison": "uncertain",
    }
    initial = consistency_report({"items": [item]})
    for raw in [{}, {"items": []}, {"items": [{"index": 0}]}, {"items": "invalid"}]:
        result = merge_review(initial, raw)
        assert result["items"] == initial["items"]
        assert result["review"]["warning"]
    result = merge_review(
        initial,
        item | {"index": 0, "expected": "不能改设定", "region_visibility": "clear", "comparison": "mismatch"},
    )
    assert result["items"][0]["status"] == "mismatch"
    assert result["items"][0]["expected"] == "红色耳饰"


def test_review_does_not_anchor_on_the_previous_verdict_or_observation():
    item = {
        "feature": "眼镜",
        "expected": "黑框眼镜",
        "support_region": "面部",
        "observation": "前次不可靠的观察",
        "region_visibility": "uncertain",
        "comparison": "uncertain",
    }
    report = consistency_report({"items": [item]})
    assert review_candidates(report) == [
        {"index": 0, "feature": "眼镜", "expected": "黑框眼镜", "support_region": "面部"}
    ]
    assert report["items"][0]["observation"] == "前次不可靠的观察"
