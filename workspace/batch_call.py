import asyncio, os, json
from openai import AsyncOpenAI

ENDPOINT = "http://localhost:8000/v1"
CONC = 1000             # == or 4× your server’s max_num_seqs

client = AsyncOpenAI(base_url=ENDPOINT, api_key="dummy", timeout=86400)

async def ask(prompt):
    resp = await client.chat.completions.create(
        model="qwen3-8b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        stream=False,
    )
    return resp.choices[0].message.content

async def main():
    prompts = ["solve this problem: " + p.strip() for p in open("/Users/htplex/Developer/ht/llm_api_wrapper/data_synced/cn_proof2.txt")]
    sem = asyncio.Semaphore(CONC)
    async def bound(p):
        async with sem:
            return await ask(p)

    tasks = [asyncio.create_task(bound(p)) for p in prompts]
    for coro in asyncio.as_completed(tasks):
        answer = await coro
        print(answer)

asyncio.run(main())