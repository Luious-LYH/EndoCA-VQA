import gzip
import hashlib
import json
from pathlib import Path

from endoca.evaluation.normalization import ParserV1


ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "core": {
        "path": ROOT / "data/manifests/endoca_core.jsonl.gz",
        "complex": 12000,
        "atomic": 15736,
        "sha256": "c603987e629300c284a8a481c9576c893257d6b0593b1e7d970f69c63a719782",
    },
    "diagnostic": {
        "path": ROOT / "data/manifests/endoca_diagnostic.jsonl.gz",
        "complex": 6000,
        "atomic": 9300,
        "sha256": "5b7057428a367dfd4c8f86a05f8d6a1e65ced80197de79fa816d1370580e9fa4",
    },
}


def _read_suite(path: Path):
    digest = hashlib.sha256()
    rows = []
    with gzip.open(path, "rb") as stream:
        for raw_line in stream:
            digest.update(raw_line)
            if raw_line.strip():
                rows.append(json.loads(raw_line))
    return rows, digest.hexdigest()


def test_manifest_counts_hashes_overlap_and_gold_parsing():
    parser = ParserV1()
    ids = {}
    for name, expected in SUITES.items():
        rows, digest = _read_suite(expected["path"])
        complex_rows = [row for row in rows if row["probe_kind"] == "complex_direct"]
        atomic_rows = [row for row in rows if row["probe_kind"] == "atomic_direct"]
        assert len(complex_rows) == expected["complex"]
        assert len(atomic_rows) == expected["atomic"]
        assert digest == expected["sha256"]
        ids[name] = {str(row["sample_id"]) for row in rows}
        for row in atomic_rows:
            parsed = parser.parse_atomic(row["gold_question_class"], row["gold_answer"])
            assert parsed["status"] == "ok"
            assert parsed["parsed"] == row["gold_parsed"]
    assert ids["core"].isdisjoint(ids["diagnostic"])
