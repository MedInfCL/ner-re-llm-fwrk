# Utilities Module

## Overview

This directory contains various utility classes and helper functions that provide supporting logic for different parts of the application. These utilities are designed to be reusable and to encapsulate specific, cross-cutting concerns such as cost management.

---

## `cost_tracker.py`

### Purpose

The `CostTracker` class is a utility designed to monitor and estimate the financial cost associated with making API calls to external LLM providers like OpenAI. It provides a simple mechanism to track token usage for each request and aggregate the total cost over a session.

### Key Features

-   **Request Logging**: The class is designed to be instantiated and passed to an LLM client. The client then calls the `log_request` method after each API call, recording the model used and the number of prompt and completion tokens.
-   **Cost Calculation**: It uses a hardcoded dictionary of prices (in USD per one million tokens) to calculate the estimated cost of each individual API request. It contains pricing for models such as "gpt-4o", "gpt-4-turbo", and "gpt-3.5-turbo".
-   **Summary and Reporting**: At the end of a prediction run, the `save_log` method can be called to generate two outputs:
    1.  A summary of the total requests, total tokens, and total estimated cost is printed to the console.
    2.  A detailed, timestamped `.csv` file is saved to the `output/logs` directory, containing a record of every API call made during the session for more granular analysis.

### Implementation notes

- Pricing units: prices in `CostTracker.model_prices` are expressed as USD per 1,000,000 tokens (input/output). The calculated per-request cost scales prompt/completion tokens accordingly.
- Unknown models: if a model name is not present in `model_prices`, the tracker will warn and treat the cost as $0 for that request.
- CSV output: `save_log()` writes a timestamped CSV (e.g., `cost_log_YYYYMMDD_HHMMSS.csv`) into `output/logs` and prints a human-readable summary to stdout. If no requests were logged, `save_log()` prints a message and does nothing.