from os.path import expanduser
import json
import sys
import os
sys.path.append(os.path.abspath('..'))
from llmw.wrapper_v2 import LLMW

    
if __name__ == "__main__":
    api_keys_path = expanduser("~/.ssh/api_keys.json")
    api_key= json.load(open(api_keys_path))["OPENAI-API-KEY"]["api_key"]
    llmw = LLMW(api_key, model="gpt-4o")
    prompts = ["solve this problem: " + p.strip() for p in open("/Users/htplex/Developer/ht/llm_api_wrapper/data_synced/cn_proof2.txt")][:200]
    answers = llmw.batch_call(prompts)
    print(len(answers))
    print(answers[0])
