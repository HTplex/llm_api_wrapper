from openai import OpenAI
import yaml
import os
from os.path import join, dirname
from pathos.pools import ProcessPool

class OpenAIWrapper:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        # TODO: Consider loading models config from a YAML file for flexibility and clarity.
        # Load model configuration from YAML file
        yaml_path = join(dirname(__file__), "models.yaml")
        with open(yaml_path, "r") as f:
            self.models = yaml.safe_load(f)

        
    
    def run(self, model, system_prompt="", user_prompt=""):
        # TODO: universal cache for all models and requests using hash
        if model not in self.models:
            raise ValueError(f"Model {model} not found")
        model_config = self.models[model]

        client = OpenAI(
            api_key=self.api_keys[model_config["api_key"]],
            base_url=model_config["base_url"]
        )
        response = client.chat.completions.create(
            model=model_config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    
    def run_job(self, job, *, max_retries: int = 10, backoff: float = 10.0):
        """
        Execute a single job with simple retry logic.

        Parameters
        ----------
        job : dict
            A job dictionary containing `model`, `system_prompt`, `prompt`, and `tag`.
        max_retries : int, optional
            How many times to retry the call if it fails. Defaults to 3.
        backoff : float, optional
            Seconds to wait between retries (multiplied exponentially). Defaults to 2.0.

        Returns
        -------
        dict
            A result dictionary with keys: tag, job, result.
        """
        import time  # Local import to keep the function picklable for ProcessPool
        attempt = 0

        while attempt <= max_retries:
            try:
                response = self.run(
                    job["model"],
                    job.get("system_prompt", ""),
                    job.get("prompt", "")
                )
                result = {
                    "tag": job["tag"],
                    "job": job,
                    "result": response
                }
                print(f"finished job: {job['tag']} (attempt {attempt + 1})")
                return result
            except Exception:
                attempt += 1
                import traceback
                print(f"Error running job {job['tag']} (attempt {attempt}):")
                traceback.print_exc()  # Print full traceback immediately

                if attempt > max_retries:
                    return {
                        "tag": job["tag"],
                        "job": job,
                        "result": None
                    }

                sleep_for = backoff * attempt
                print(f"Retrying job {job['tag']} in {sleep_for:.1f}s...")
                time.sleep(sleep_for)
    def run_batch(self, jobs, num_workers=16):
        pool = ProcessPool(num_workers)
        results = pool.map(self.run_job, jobs)
        pool.close()
        pool.join()
        return results