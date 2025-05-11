from os.path import join, expanduser
import json
from pprint import pprint
from llmw.wrappers import OpenAIWrapper

API_KEY_JSON_PATH = expanduser("~/.ssh/api_keys.json")
with open(API_KEY_JSON_PATH, 'r') as fp:
    api_keys = json.load(fp)
api_keys = {"openai": api_keys["OPENAI-API-KEY"]["api_key"],
            "openrouter": api_keys["OPENROUTER-API-KEY"]["api_key"],
            "deepseek": api_keys["DEEPSEEK-API-KEY"]["api_key"]}


ow = OpenAIWrapper(api_keys)


from pathos.multiprocessing import ProcessingPool
from tqdm import tqdm
import json

with open("./tmp/human_check_gen.json") as fp:
    gens = json.load(fp)
with open("./tmp/human_check_solve.json") as fp:
    solves = json.load(fp)

# Define the list of models to iterate over for each generation
job_models = [
    "gpt-4o",
    "deepseek-r1",
    "qwen3-8b",
    "glm-z1-9b",
    "deepseek-r1d-llama-8b"
]
# Build the jobs list by iterating over each generation and model combination
jobs = []
for idx, gen in enumerate(gens):
    for model in job_models:
        jobs.append({
            "tag": "{}:{}:{}".format("gen",model,idx),
            "model": model,
            "system_prompt": "你是一个助手。",
            "prompt": gen["question"]
        })
for idx, solve in enumerate(solves):
    for model in job_models:
        jobs.append({
            "tag": "{}:{}:{}".format("solve",model,idx),
            "model": model,
            "system_prompt": "你是一个助手。",
            "prompt": solve["question"]
        })


with ProcessingPool(nodes=16) as pool:
    results = list(tqdm(pool.imap(ow.run_job, jobs), total=len(jobs)))
with open("./tmp/gen_results.json", 'w') as fp:
    json.dump(results, fp, sort_keys=True, indent=4, ensure_ascii=False)