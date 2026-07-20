import pytest

from endoca.evaluation.normalization import ParserV1
from endoca.evaluation.score import coerce_prediction, score_atomic


@pytest.fixture
def parser() -> ParserV1:
    return ParserV1()


@pytest.mark.parametrize(
    ("question_class", "answer", "expected"),
    [
        ("text_presence", "Visible", "yes"),
        ("box_artifact_presence", "not visible", "no"),
        ("polyp_count", "two", "2"),
        ("procedure_type", "colonoscopy", "colonoscopy"),
        ("polyp_type", "sessile and flat", "flat;sessile"),
        ("abnormality_color", "red, pink", "pink;red"),
        ("instrument_location", "lower-rigth and center", "center;lower-right"),
    ],
)
def test_parser_supported_answer_types(parser, question_class, answer, expected):
    result = parser.parse_atomic(question_class, answer)
    assert result["status"] == "ok"
    assert result["parsed"] == expected


@pytest.mark.parametrize("answer", ["", "maybe", "uncertain"])
def test_binary_empty_or_ambiguous_is_not_parseable(parser, answer):
    result = parser.parse_atomic("text_presence", answer)
    assert result["status"] == "unparseable"


def test_verbose_count_is_coerced_before_scoring(parser):
    row = {
        "gold_question_class": "polyp_count",
        "gold_parsed": "1",
        "prediction": "The answer is one.",
        "error": "",
    }
    result = score_atomic(row, parser)
    assert coerce_prediction("polyp_count", row["prediction"]) == "1"
    assert result["is_correct"] is True


def test_empty_prediction_is_incorrect(parser):
    result = score_atomic(
        {
            "gold_question_class": "text_presence",
            "gold_parsed": "yes",
            "prediction": "",
            "error": "",
        },
        parser,
    )
    assert result["parse_status"] == "failure"
    assert result["is_correct"] is False
