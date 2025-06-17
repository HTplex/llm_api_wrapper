import asyncio, os, json
from openai import AsyncOpenAI


class OpenAIBatchWrapper:
    def __init__(self, endpoint: str, api_key: str, model: str, max_tokens: int, max_concurrency: int, system_prompt: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.max_concurrency = max_concurrency
        self.client = AsyncOpenAI(base_url=endpoint, api_key=api_key, timeout=86400)
        self.system_prompt = system_prompt
        
    
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
        
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=prompt_full,
            max_tokens=self.max_tokens,
            stream=False,
        )
        return resp.choices[0].message.content
    
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
    
if __name__ == "__main__":
    client = OpenAIBatchWrapper(
        endpoint="http://localhost:8000/v1",
        api_key="dummy",
        model="qwen3-8b",
        max_tokens=4096,
        max_concurrency=1000,
        system_prompt="You are a helpful assistant."
    )
    prompts = ["solve this problem: " + p.strip() for p in open("/Users/htplex/Developer/ht/llm_api_wrapper/data_synced/cn_proof2.txt")]
    print(len(prompts))
    print(prompts[0])
    answers = asyncio.run(client.batch_call(prompts))
    print(len(answers))
    print(answers[0])