"""Question-class-aware answer normalization used by the EndoCA scorer."""

from __future__ import annotations

import re


class ParserV1:
    """Conservatively normalize EndoCA atomic answers."""

    def __init__(self) -> None:
        self.parser_version = "v1"
        self.negative_variants = {
            "none",
            "no",
            "not relevant",
            "not applicable",
            "n/a",
            "not present",
            "absent",
            "nothing",
            "0",
            "no text",
            "not visible",
            "not seen",
            "not identified",
        }
        self.affirmative_variants = {"yes", "present", "visible", "seen", "identified"}
        self.number_words = {
            "zero": 0,
            "none": 0,
            "no": 0,
            "one": 1,
            "single": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
        }
        self.token_aliases = {
            "lower-rigth": "lower-right",
            "center-rigth": "center-right",
            "upper-rigth": "upper-right",
        }

    @staticmethod
    def normalize_answer(raw_answer: str) -> str:
        if not raw_answer:
            return ""
        normalized = re.sub(r"\s+", " ", str(raw_answer).strip().lower())
        return normalized.strip(" .,:;!?")

    def parse_set_answer(self, raw_answer: str) -> list[str]:
        normalized = self.normalize_answer(raw_answer)
        if not normalized:
            return []
        items = [part.strip() for part in re.split(r"\s*(?:;|,|\band\b)\s*", normalized)]
        return [self.token_aliases.get(item, item) for item in items if item]

    def is_negative(self, raw_answer: str) -> bool:
        return self.normalize_answer(raw_answer) in self.negative_variants

    def parse_boolean(self, raw_answer: str, label: str) -> dict:
        normalized = self.normalize_answer(raw_answer)
        if normalized in self.affirmative_variants:
            return {"parsed": "yes", "status": "ok", "reason": None}
        if self.is_negative(normalized):
            return {"parsed": "no", "status": "ok", "reason": None}
        return {
            "parsed": normalized,
            "status": "unparseable",
            "reason": f"non-boolean {label}: {raw_answer}",
        }

    def parse_count(self, raw_answer: str, label: str) -> dict:
        normalized = self.normalize_answer(raw_answer)
        if normalized in self.number_words:
            return {"parsed": str(self.number_words[normalized]), "status": "ok", "reason": None}
        try:
            count = int(normalized)
        except ValueError:
            return {
                "parsed": normalized,
                "status": "unparseable",
                "reason": f"non-integer {label}: {raw_answer}",
            }
        return {"parsed": str(count), "status": "ok", "reason": None}

    def parse_polyp_removal_status(self, raw_answer: str) -> dict:
        normalized = self.normalize_answer(raw_answer)
        if normalized in {"yes", "all removed", "complete"}:
            return {"parsed": "yes", "status": "ok", "reason": None}
        if normalized in {"no", "incomplete", "partial"}:
            return {"parsed": "no", "status": "ok", "reason": None}
        if normalized in {"not relevant", "not applicable", "n/a"}:
            return {"parsed": "not_applicable", "status": "ok", "reason": None}
        return {
            "parsed": normalized,
            "status": "unparseable",
            "reason": f"unknown removal status: {raw_answer}",
        }

    def parse_location(self, raw_answer: str) -> dict:
        if self.is_negative(raw_answer):
            return {"parsed": "none", "status": "ok", "reason": None}
        locations = self.parse_set_answer(raw_answer)
        if not locations:
            return {"parsed": "", "status": "unparseable", "reason": "empty location"}
        return {"parsed": ";".join(sorted(locations)), "status": "ok", "reason": None}

    def parse_color(self, raw_answer: str) -> dict:
        if self.is_negative(raw_answer):
            return {"parsed": "none", "status": "ok", "reason": None}
        colors = self.parse_set_answer(raw_answer)
        if not colors:
            return {"parsed": "", "status": "unparseable", "reason": "empty color"}
        return {"parsed": ";".join(sorted(colors)), "status": "ok", "reason": None}

    def parse_presence(self, raw_answer: str, entity_type: str) -> dict:
        normalized = self.normalize_answer(raw_answer)
        if normalized in {"yes", "present"}:
            return {"parsed": "yes", "status": "ok", "reason": None}
        if normalized in {"no", "none", "absent", "not present"}:
            return {"parsed": "no", "status": "ok", "reason": None}
        items = self.parse_set_answer(raw_answer)
        if items:
            return {"parsed": ";".join(sorted(items)), "status": "ok", "reason": None}
        return {
            "parsed": normalized,
            "status": "unparseable",
            "reason": f"ambiguous {entity_type} presence: {raw_answer}",
        }

    def parse_procedure_type(self, raw_answer: str) -> dict:
        normalized = self.normalize_answer(raw_answer)
        known = {"colonoscopy", "gastroscopy", "endoscopy", "upper endoscopy", "lower endoscopy"}
        if normalized in known:
            return {"parsed": normalized, "status": "ok", "reason": None}
        if self.is_negative(normalized):
            return {"parsed": "unknown", "status": "ok", "reason": None}
        return {
            "parsed": normalized,
            "status": "partial",
            "reason": f"unknown procedure type: {raw_answer}",
        }

    def parse_polyp_type(self, raw_answer: str) -> dict:
        normalized = self.normalize_answer(raw_answer)
        if self.is_negative(normalized) or "no polypoid" in normalized:
            return {"parsed": "none", "status": "ok", "reason": None}
        items = self.parse_set_answer(raw_answer)
        known = {
            "sessile",
            "pedunculated",
            "flat",
            "adenomatous",
            "hyperplastic",
            "serrated",
            "paris ip",
            "paris is",
            "paris iia",
        }
        if items and all(item in known for item in items):
            return {"parsed": ";".join(sorted(items)), "status": "ok", "reason": None}
        return {
            "parsed": normalized,
            "status": "unparseable",
            "reason": f"unknown polyp type: {raw_answer}",
        }

    def parse_size(self, raw_answer: str, label: str) -> dict:
        normalized = self.normalize_answer(raw_answer).replace(" ", "")
        if self.is_negative(normalized):
            return {"parsed": "none", "status": "ok", "reason": None}
        items = [">20mm" if item == ">20" else item for item in self.parse_set_answer(normalized)]
        known = {"<5mm", "5-10mm", "11-20mm", ">20mm"}
        if items and all(item in known for item in items):
            return {"parsed": ";".join(sorted(items)), "status": "ok", "reason": None}
        return {
            "parsed": normalized,
            "status": "unparseable",
            "reason": f"unknown {label}: {raw_answer}",
        }

    def parse_atomic(self, question_class: str, raw_answer: str) -> dict:
        if question_class == "polyp_count":
            result = self.parse_count(raw_answer, "polyp count")
        elif question_class in {"instrument_count", "finding_count"}:
            result = self.parse_count(raw_answer, question_class)
        elif question_class == "polyp_removal_status":
            result = self.parse_polyp_removal_status(raw_answer)
        elif question_class in {"abnormality_location", "instrument_location", "landmark_location"}:
            result = self.parse_location(raw_answer)
        elif question_class in {"abnormality_color", "landmark_color"}:
            result = self.parse_color(raw_answer)
        elif question_class in {"text_presence", "box_artifact_presence"}:
            result = self.parse_boolean(raw_answer, question_class)
        elif question_class in {"abnormality_presence", "instrument_presence", "landmark_presence", "finding_presence"}:
            result = self.parse_presence(raw_answer, question_class.removesuffix("_presence"))
        elif question_class == "procedure_type":
            result = self.parse_procedure_type(raw_answer)
        elif question_class == "polyp_type":
            result = self.parse_polyp_type(raw_answer)
        elif question_class == "polyp_size":
            result = self.parse_size(raw_answer, "polyp_size")
        else:
            result = {
                "parsed": self.normalize_answer(raw_answer),
                "status": "unparseable",
                "reason": f"uncovered question_class: {question_class}",
            }
        return {
            "question_class": question_class,
            "raw_answer": raw_answer,
            "parse_status": result["status"],
            **result,
        }
