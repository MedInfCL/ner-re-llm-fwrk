import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict

class CostTracker:
    """
    A utility class to track token usage and estimate costs for LLM API calls.

    This class logs each request, calculates its cost based on a predefined
    price list, and can save a detailed report to a CSV file.
    """

    def __init__(self, log_dir: str = "output/logs"):
        """
        Initializes the CostTracker.

        Args:
            log_dir (str): The directory where the cost tracking log will be saved.
        """
        self.requests_log: List[Dict] = []
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Pricing is in dollars per 1 million tokens (input/output)
        self.model_prices = {
            "gpt-4o": {"input": 5.00, "output": 15.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50}
        }

    def log_request(self, model: str, prompt_tokens: int, completion_tokens: int):
        """
        Logs the details of a single API request and calculates its cost.

        Args:
            model (str): The name of the model used for the request.
            prompt_tokens (int): The number of tokens in the input prompt.
            completion_tokens (int): The number of tokens in the generated output.
        """
        if model not in self.model_prices:
            print(f"Warning: Model '{model}' not found in price list. Cost will be calculated as $0.")
            prices = {"input": 0, "output": 0}
        else:
            prices = self.model_prices[model]

        input_cost = (prompt_tokens / 1_000_000) * prices["input"]
        output_cost = (completion_tokens / 1_000_000) * prices["output"]
        total_cost = input_cost + output_cost

        self.requests_log.append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_usd": total_cost
        })

    def get_summary(self) -> Dict:
        """
        Calculates and returns a summary of all logged requests.

        Returns:
            Dict: A dictionary containing the total number of requests,
                  total tokens used, and the total estimated cost.
        """
        total_requests = len(self.requests_log)
        total_tokens = sum(req['total_tokens'] for req in self.requests_log)
        total_cost = sum(req['estimated_cost_usd'] for req in self.requests_log)

        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_estimated_cost_usd": total_cost
        }

    def save_log(self):
        """
        Saves the detailed request log to a timestamped CSV file and prints a summary.
        """
        if not self.requests_log:
            print("No requests were logged. Nothing to save.")
            return

        # Create a DataFrame for easy CSV export
        log_df = pd.DataFrame(self.requests_log)

        # Generate a unique filename with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"cost_log_{timestamp}.csv"

        log_df.to_csv(log_path, index=False)
        print(f"\n--- API Cost & Usage Summary ---")
        summary = self.get_summary()
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value:,.2f}" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}")
        
        print(f"\nDetailed cost log saved to: {log_path}")