import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_benchmark_tables_have_eleven_models():
    for suite in ("endoca_core", "endoca_diagnostic"):
        rows = _rows(f"results/paper/benchmark/{suite}_results.csv")
        assert len(rows) == 11
        assert all(int(row["qa_count"]) in {27736, 15300} for row in rows)


def test_asr_macro_values_match_paper_table():
    rows = _rows("results/paper/asr/asr_results.csv")
    average = next(row for row in rows if row["model"] == "Average")
    assert len(rows) == 5
    assert average["direct_complex_accuracy"] == "63.3"
    assert average["revise_complex_accuracy"] == "62.2"
    assert average["joint_accuracy_delta"] == "5.4"
    assert average["inconsistency_delta"] == "-10.4"
    assert average["selective_coverage"] == "75.9"
    assert average["selective_complex_accuracy"] == "71.0"


def test_no_atomic_control_values_match_paper_table():
    rows = _rows("results/paper/ablation/no_atomic_ablation.csv")
    asr = next(row for row in rows if row["method"] == "ASR")
    assert len(rows) == 3
    assert asr["joint_acc_percent"] == "50.9"
    assert asr["complex_atomic_inconsistency_percent"] == "17.1"
    assert asr["coverage_percent"] == "75.9"
