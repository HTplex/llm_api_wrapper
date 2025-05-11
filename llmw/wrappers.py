from openai import OpenAI
import yaml
import os
from os.path import join, dirname

class OpenAIWrapper:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        # TODO: Consider loading models config from a YAML file for flexibility and clarity.
        # Load model configuration from YAML file
        yaml_path = join(dirname(__file__), "models.yaml")
        with open(yaml_path, "r") as f:
            self.models = yaml.safe_load(f)

        
    
    def run(self, model, system_prompt="", user_prompt=""):
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
    
    def run_job(self,job):
        try:
            result = {
                "tag": job["tag"],
                "result": self.run(job["model"], job["system_prompt"], job["prompt"])
            }
            print("finished job: {}".format(job["tag"]))
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {
                "tag": job["tag"],
                "result": None
            }
        return result