import os
import json
from openai import OpenAI, APIError, APITimeoutError
from typing import List, Dict, Any, Optional

# Add the project root to the Python path to allow for absolute imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm_services.base_client import BaseLLMClient

class OpenAIClient(BaseLLMClient):
    """
    A concrete implementation of the BaseLLMClient for interacting with
    the OpenAI API (e.g., GPT-4, GPT-3.5).
    """

    def __init__(self, config: Dict[str, Any], api_key: str = None):
        """
        Initializes the OpenAI client.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration
                                     for the OpenAI provider, including 'model'
                                     and 'temperature'.
            api_key (str, optional): The OpenAI API key. If not provided, it will
                                     be read from the OPENAI_API_KEY environment
                                     variable.
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided or set as an environment variable (OPENAI_API_KEY).")
        
        self.client = OpenAI(api_key=api_key)
        self.model = config.get("model", "gpt-4o")
        self.temperature = config.get("temperature", 0.1)

        request_settings = config.get("request_settings", {})
        self.max_retries = request_settings.get("max_retries", 3)
        self.initial_timeout = request_settings.get("initial_timeout_seconds", 20)
        self.backoff_factor = request_settings.get("backoff_factor", 1.5)

    def get_ner_prediction(self, prompt: str, trace: Optional[Any] = None, report_index: int = -1) -> List[Dict[str, Any]]:
        """
        Sends a prompt to the OpenAI API and returns the extracted entities.

        If a trace object is provided, this method will log the API call
        as a generation nested within that trace.

        Args:
            prompt (str): The fully constructed prompt for the NER task.
            trace (Optional[Any]): The Langfuse trace object.
            report_index (int, optional): The index of the report in the test set for tracing.

        Returns:
            List[Dict[str, Any]]: A list of extracted entity dictionaries.
                                  Returns an empty list in case of an API error
                                  or if the response is not valid JSON.
        """
        generation = None
        response_content = ""
        current_timeout = self.initial_timeout

        for attempt in range(self.max_retries + 1):
            try:
                if trace:
                    with trace.start_as_current_generation(
                        name=f"NER Prediction (Report {report_index}, Attempt {attempt + 1})" if report_index != -1 else f"NER Prediction (Attempt {attempt + 1})",
                        model=self.model,
                        input=prompt,
                        metadata={"temperature": self.temperature, "timeout": current_timeout}
                    ) as generation:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            temperature=self.temperature,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant designed to return JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            response_format={"type": "json_object"},
                            timeout=current_timeout
                        )
                        response_content = response.choices[0].message.content
                        generation.update(output=response_content, usage=response.usage)
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant designed to return JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        timeout=current_timeout
                    )
                    response_content = response.choices[0].message.content

                response_dict = json.loads(response_content)
                
                for value in response_dict.values():
                    if isinstance(value, list):
                        return value
                
                return [] # Return empty if JSON is valid but doesn't contain a list

            except APITimeoutError:
                error_message = f"OpenAI API request timed out after {current_timeout:.2f} seconds. Retrying... (Attempt {attempt + 2}/{self.max_retries + 1})"
                print(f"Warning: {error_message}")
                if generation:
                    generation.update(level='WARNING', status_message=error_message)
                
                # Increase timeout for the next attempt
                current_timeout *= self.backoff_factor
                continue # Go to the next iteration of the loop

            except APIError as e:
                error_message = f"OpenAI API returned an error: {e}"
                print(f"Error: {error_message}")
                if generation:
                    generation.update(level='ERROR', status_message=error_message)
                return []
            except json.JSONDecodeError:
                error_message = f"Failed to decode JSON from model response: {response_content}"
                print(f"Error: {error_message}")
                if generation:
                    generation.update(level='ERROR', status_message=error_message)
                return []
            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"
                print(f"Error: {error_message}")
                if generation:
                    generation.update(level='ERROR', status_message=error_message)
                return []

        print(f"Error: Request failed after {self.max_retries + 1} attempts.")
        return []
    

    def get_re_prediction(self, prompt: str, trace: Optional[Any] = None, report_index: int = -1) -> List[Dict[str, Any]]:
        """
        Sends a prompt to the OpenAI API and returns the extracted relations.

        Args:
            prompt (str): The fully constructed prompt for the RE task.
            trace (Optional[Any]): The Langfuse trace object.
            report_index (int, optional): The index of the report in the test set for tracing.

        Returns:
            List[Dict[str, Any]]: A list of extracted relation dictionaries.
                                  Returns an empty list in case of an API error.
        """
        generation = None
        response_content = ""
        current_timeout = self.initial_timeout

        for attempt in range(self.max_retries + 1):
            try:
                if trace:
                    with trace.start_as_current_generation(
                        name=f"RE Prediction (Report {report_index}, Attempt {attempt + 1})" if report_index != -1 else f"RE Prediction (Attempt {attempt + 1})",
                        model=self.model,
                        input=prompt,
                        metadata={"temperature": self.temperature, "timeout": current_timeout}
                    ) as generation:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            temperature=self.temperature,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant designed to return JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            response_format={"type": "json_object"},
                            timeout=current_timeout
                        )
                        response_content = response.choices[0].message.content
                        generation.update(output=response_content, usage=response.usage)
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant designed to return JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        timeout=current_timeout
                    )
                    response_content = response.choices[0].message.content

                response_dict = json.loads(response_content)
                
                for value in response_dict.values():
                    if isinstance(value, list):
                        return value
                
                return []

            except APITimeoutError:
                error_message = f"OpenAI API request timed out after {current_timeout:.2f} seconds. Retrying... (Attempt {attempt + 2}/{self.max_retries + 1})"
                print(f"Warning: {error_message}")
                if generation:
                    generation.update(level='WARNING', status_message=error_message)
                
                current_timeout *= self.backoff_factor
                continue

            except APIError as e:
                error_message = f"OpenAI API returned an error: {e}"
                print(f"Error: {error_message}")
                if generation:
                    generation.update(level='ERROR', status_message=error_message)
                return []
            except json.JSONDecodeError:
                error_message = f"Failed to decode JSON from model response: {response_content}"
                print(f"Error: {error_message}")
                if generation:
                    generation.update(level='ERROR', status_message=error_message)
                return []
            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"
                print(f"Error: {error_message}")
                if generation:
                    generation.update(level='ERROR', status_message=error_message)
                return []

        print(f"Error: Request failed after {self.max_retries + 1} attempts.")
        return []