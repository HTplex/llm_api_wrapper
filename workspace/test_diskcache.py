from diskcache import Cache
import hashlib, json
from time import sleep

cache_dir = "/tmp/test_cache"

cache = Cache(cache_dir,size_limit=1e11,compress_level=3) # 100GB      # any directory

def cache_lookup(prompt):
    hash_key = hashlib.sha256(prompt.encode()).hexdigest()
    if hash_key in cache:
        return cache[hash_key]              # hit
    value = {"prompt": prompt, "response": "test"}
    sleep(10)
    cache.set(hash_key, value,  tag="llm")   # TTL = 1 day
    return value

print(cache_lookup("test"))
print(cache_lookup("test"))
print(cache_lookup("test"))

