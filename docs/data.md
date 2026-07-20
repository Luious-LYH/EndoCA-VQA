# Data Guide

## Manifest Rows

Every JSONL row identifies one image-question query. `probe_kind` is `complex_direct` for a complex question and `atomic_direct` for one associated atomic question. Rows sharing a `sample_id` belong to the same paired evaluation unit.

Important fields are `sample_id`, `probe_id`, `image`, `question`, `gold_answer`, `question_class`, `complexity`, and `probe_kind`. Complex rows also contain `gold_atoms`; atomic rows contain `atomic_index`, `gold_question_class`, and `gold_parsed`.

## Prediction Rows

A model output should retain the manifest row and add:

```json
{
  "model_id": "your/model",
  "prediction": "model answer",
  "error": ""
}
```

The scorer does not use an additional LLM judge. It normalizes concise answer forms with deterministic rules for binary, count, categorical, and multi-label questions.

## Metrics

- **Complex accuracy:** complex predictions matching every associated answer component.
- **Atomic accuracy:** correctness averaged over separately generated atomic answers.
- **Joint accuracy:** both the complex answer and every associated atomic answer are correct for a sample.
- **Complex-atomic inconsistency:** among correct complex answers, the fraction with at least one wrong atomic answer.

The image and source annotations originate from Kvasir-VQA-x1. See [`NOTICE`](../NOTICE) before redistributing data.
