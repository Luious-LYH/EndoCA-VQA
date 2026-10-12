import pytest

from endoca.evaluation.score import summarize


def _complex(sample_id: int, correct: bool) -> dict:
    return {
        "sample_id": sample_id,
        "model_id": "test/model",
        "probe_kind": "complex_direct",
        "is_correct": correct,
        "parse_status": "ok",
        "complexity": 1,
    }


def _atomic(sample_id: int, correct: bool) -> dict:
    return {
        "sample_id": sample_id,
        "model_id": "test/model",
        "probe_kind": "atomic_direct",
        "is_correct": correct,
        "parse_status": "ok",
        "complexity": 1,
        "gold_question_class": "text_presence",
    }


def test_joint_accuracy_and_inconsistency():
    rows = [
        _complex(1, True),
        _atomic(1, True),
        _complex(2, True),
        _atomic(2, False),
        _complex(3, False),
        _atomic(3, True),
    ]
    metrics = summarize(rows)["test/model"]
    assert metrics["complex_accuracy"] == pytest.approx(2 / 3)
    assert metrics["atomic_accuracy"] == pytest.approx(2 / 3)
    assert metrics["joint_accuracy"] == pytest.approx(1 / 3)
    assert metrics["complex_atomic_inconsistency"] == pytest.approx(1 / 2)
