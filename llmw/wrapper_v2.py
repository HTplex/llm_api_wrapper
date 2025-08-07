import hashlib
import asyncio, os, json
from openai import AsyncOpenAI
import yaml
from os.path import expanduser
from diskcache import Cache
from os.path import join, dirname, abspath

class LLMW:
    def __init__(self,
                 api_key,
                 model,
                 cache_dir = expanduser("~/data/llm_cache"),
                 max_tokens=16384,
                 max_concurrency=100,
                 system_prompt="You are a helpful assistant."):
        models_path = join(dirname(abspath(__file__)), "models.yaml")
        with open(models_path, "r") as f:
            self.configs = yaml.safe_load(f)
        self.cache_dir = cache_dir
        self.model = model
        self.client = OpenAIBatchWrapper(
            endpoint=self.configs[model]["base_url"],
            api_key=api_key,
            model=self.configs[model]["model"],
            max_tokens=max_tokens,
            max_concurrency=max_concurrency,
            system_prompt=system_prompt,
            cache_dir=cache_dir)
    
    def batch_call(self, prompts):
        return asyncio.run(self.client.batch_call(prompts))
    
class OpenAIBatchWrapper:
    def __init__(self, endpoint: str,
                 api_key: str,
                 model: str,
                 max_tokens: int,
                 max_concurrency: int,
                 system_prompt: str,
                 cache_dir: str,
                 cache_tag = None):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.max_concurrency = max_concurrency
        self.client = AsyncOpenAI(base_url=endpoint, api_key=api_key, timeout=86400)
        self.system_prompt = system_prompt
        self.cache = Cache(cache_dir,size_limit=1e11,compress_level=3) # 100GB      # any directory
        if not cache_tag:
            self.cache_tag = f"llm:{model}"
        else:
            self.cache_tag = cache_tag

        
    
    async def call(self, prompt):
        if isinstance(prompt, dict):
            prompt_full = prompt.copy()
        elif isinstance(prompt, str):
            if self.system_prompt:
                prompt_full = [{"role": "system", "content": self.system_prompt},
                               {"role": "user", "content": prompt}]
            else:
                prompt_full = [{"role": "user", "content": prompt}]
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")
        
        hash_key = hashlib.sha256((json.dumps(prompt_full)+self.model).encode()).hexdigest()
        if hash_key in self.cache:
            return self.cache[hash_key]["response"]
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=prompt_full,
                max_tokens=self.max_tokens,
                stream=False,
            )
            resp = resp.choices[0].message.content
            
            self.cache.set(hash_key, {"prompt": prompt_full,
                                    "response": resp}, tag=self.cache_tag)
        except Exception as e:
            print(f"Error: {e}")
            return None
        return resp
    
    async def batch_call(self, prompts: 'list[dict] | list[str]'):
        """
        Execute a batch of prompts concurrently while reporting progress.

        The returned list keeps the same ordering as the supplied `prompts`.
        Progress is printed roughly ten times throughout the run (at least once).
        """
        total = len(prompts)
        sem = asyncio.Semaphore(self.max_concurrency)

        async def bound(idx: int, prompt):
            """Wrap `self.call` so we can restore original ordering."""
            async with sem:
                result = await self.call(prompt)
                return idx, result

        # Launch all tasks
        tasks = [asyncio.create_task(bound(i, p)) for i, p in enumerate(prompts)]

        results = [None] * total
        completed = 0

        for coro in asyncio.as_completed(tasks):
            idx, content = await coro
            results[idx] = content
            completed += 1

            print(f"[batch_call] Progress: {completed}/{total} completed")

        return results