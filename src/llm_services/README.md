# LLM Services Module

## `__init__.py`

### Purpose

This file contains the `get_llm_client` factory function, which serves as the single entry point for the rest of the application to obtain an LLM client.

### Key features

- Configuration-driven: the factory inspects the provided RAG configuration (the application calls `get_llm_client(config_path=...)`) to determine which provider to instantiate. The config keys used include `llm.provider` and provider-specific settings under `llm`.
- Dynamic instantiation: based on the provider string it returns the corresponding concrete client (`OpenAIClient`, `OllamaClient`, etc.), decoupling caller code from provider-specific details.

Usage note: the factory expects the caller to pass the path to a YAML config file; the provided `scripts/evaluation/generate_rag_predictions.py` is the usual caller.

---

## `base_client.py`

### Purpose

This file defines the `BaseLLMClient` abstract base class. It serves as the contract that all concrete LLM clients must implement.

### Interface contract

- `get_ner_prediction(prompt: str, trace: Optional[Any] = None, report_index: int = -1) -> List[Dict[str, Any]]]`
	- Should return a list of entity dictionaries (for example: `{"text": "nódulo", "label": "HALL", "start_offset": 0, "end_offset": 6}`).
- `get_re_prediction(prompt: str, trace: Optional[Any] = None, report_index: int = -1) -> List[Dict[str, Any]]]`
	- Should return a list of relation dictionaries (for example: `{"from_id": 1, "to_id": 2, "type": "ubicar"}`).

Both methods should return an empty list on errors or when nothing relevant is found. Implementations may use the `trace` object (Langfuse) to record generation metadata.

---

## `openai_client.py`

### Purpose

`OpenAIClient` is the concrete implementation for calling OpenAI via the `openai` Python package (wrapped here as `OpenAI`).

### Important details

- Authentication: the client requires an API key, which is read from the `OPENAI_API_KEY` environment variable if not passed explicitly to the constructor. The constructor will raise a `ValueError` if no key is provided.
- Request format: the client sends a chat-completion request and asks for a JSON object response (`response_format={'type': 'json_object'}`). It then parses `response.choices[0].message.content` as JSON and returns the first list found inside the parsed object.
- Retries and timeouts: retry logic is controlled by `config['request_settings']` (keys: `max_retries`, `initial_timeout_seconds`, `backoff_factor`). Timeouts raise `APITimeoutError` and trigger exponential backoff.
- Observability: when a Langfuse `trace` object is provided, each call is logged as a generation with `input`, `output`, and `usage` metadata.
- Error handling: network/API errors and JSON parsing errors are caught and cause the method to return an empty list.

Usage tip: ensure the OpenAI model chosen in the config supports the JSON response format used here.

---

## `ollama_client.py`

### Purpose

`OllamaClient` implements `BaseLLMClient` for local Ollama servers (via `langchain_community.llms.Ollama`).

### Important details

- Initialization: the client accepts `model`, `base_url` (defaults to `http://localhost:11434`) and `llm_parameters` in its config and constructs an `Ollama` client.
- Response parsing: Ollama responses may contain conversational text. The client searches the response for the first JSON array or object using a regex and attempts to parse and return the list found. If the parsed JSON is an object, the client searches its values for a list; otherwise it logs a warning and returns an empty list.
- Observability: if a `trace` is provided the raw response and parsed JSON are recorded in the generation for analysis.

Usage tip: ensure your Ollama instance is reachable and configured to return JSON-like output if you rely on automatic JSON extraction.

---

## Available clients

- `OpenAIClient` — uses the OpenAI API. Requires `OPENAI_API_KEY` env var or passing an `api_key` to the constructor. Supports retries, timeouts, and Langfuse tracing.
- `OllamaClient` — connects to a local Ollama server. No secret env var required but ensure `base_url` and `model` are set correctly in the config.

If you add other providers, implement the methods from `BaseLLMClient` and update the factory in `__init__.py` accordingly.
