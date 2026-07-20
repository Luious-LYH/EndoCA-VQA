# Model Usage

`configs/models.yaml` records the model identifiers and adapter routes used in the paper. It is an example registry rather than a universal environment specification.

Use `--config configs/models.yaml --model MODEL_KEY` for a listed open model, or pass `--model-id` and `--adapter` directly for another model. Optional local PEFT and assembled LLaVA paths are supplied through `--adapter-path` and `--llava-model-path`. Add `--offline` when every required checkpoint is already cached.

API inference supports OpenAI-compatible and Anthropic-compatible endpoints. Credentials are read only from the environment variable named by `--api-key-env`; do not place keys in YAML, scripts, or command history.
