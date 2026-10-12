import pytest

from endoca.asr.build import write_jsonl
from endoca.asr.run import normalize_status, parse_json_object


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("fully supported", "fully_supported"),
        ("partial", "partially_supported"),
        ("conflicting", "insufficient_or_conflicting"),
        ("unexpected", "insufficient_or_conflicting"),
    ],
)
def test_asr_status_normalization(raw, expected):
    assert normalize_status(raw) == expected


def test_asr_json_parsing_accepts_fenced_output():
    parsed = parse_json_object(
        '```json\n{"revised_answer":"yes","support_status":"fully_supported","selective_answer":"yes"}\n```'
    )
    assert parsed["revised_answer"] == "yes"


def test_asr_manifest_rejects_gold_fields(tmp_path):
    with pytest.raises(ValueError, match="leaked"):
        write_jsonl([{"sample_id": 1, "gold_answer": "yes"}], tmp_path / "unsafe.jsonl")
