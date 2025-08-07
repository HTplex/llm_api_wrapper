# run embeddings
import hashlib
from pathlib import Path
from typing import List, Optional

import numpy as np
from diskcache import Cache
from openai import OpenAI
from typing import List, Optional, TypedDict, Dict

"""
# start a vllm server

python -m vllm.entrypoints.openai.api_server \
    --model               bge-m3 \
    --served-model-name   "bge-m3" \
    --trust-remote-code \
    --dtype               bfloat16 \
    --host                0.0.0.0 \
    --port                8946 \
    --gpu-memory-utilization 0.92 \
    --max-model-len       1024 \
    --max-num-batched-tokens 819200 \
    --max-num-seqs        8000 \
    --enable-prefix-caching \
    --block-size          16 \
    --disable-log-requests 
"""


class EmbeddingRecord(TypedDict):
    """Cache/lookup payload for a single text string."""

    text: str
    embedding: Optional[np.ndarray]


# ---------------------------------------------------------------------------
# Core embedding call
# ---------------------------------------------------------------------------

def run_embeddings(
    text_list: List[str],
    *,
    base_url: str = "http://localhost:8946",  # *without* the /v1 suffix
    model: str = "bge-m3",
) -> List[np.ndarray]:
    """Return `np.ndarray` embeddings for every item in *text_list*.

    This is a thin wrapper around a vLLM‑compatible OpenAI embedding endpoint.
    """

    if not text_list:
        return []

    client = OpenAI(base_url=f"{base_url}/v1", api_key="local-vllm")

    resp = client.embeddings.create(
        model=model,
        input=text_list,
        encoding_format="float",  # switch to "base64" for smaller payloads
    )

    return [np.asarray(item.embedding, dtype=np.float32) for item in resp.data]


# ---------------------------------------------------------------------------
# Cached variant
# ---------------------------------------------------------------------------

def run_embeddings_with_cache(
    text_list: List[str],
    *,
    base_url: str = "http://localhost:8946",
    model: str = "bge-m3",
    cache_path: Optional[str | Path] = None,
    overwrite_cache: bool = False,
    cache_size_limit: float = 1e11,  # 100 GB
) -> List[np.ndarray]:
    """As :func:`run_embeddings` **plus** on‑disk caching via *diskcache*.

    The cache key is *sha256(text)*.  Values conform to :class:`EmbeddingRecord`.
    Type‑checkers no longer complain because the schema explicitly admits *str* + *ndarray*.
    """

    # Build an ordered mapping text‑hash → record skeleton.
    hashed: Dict[str, EmbeddingRecord] = {}
    for txt in text_list:
        key = hashlib.sha256(txt.encode()).hexdigest()
        hashed[key] = {"text": txt, "embedding": None}

    # Optional: hydrate from existing cache on disk.
    cache: Optional[Cache] = None
    if cache_path is not None:
        cache = Cache(str(cache_path), size_limit=cache_size_limit, compress_level=3)
        if not overwrite_cache:
            for key in hashed:
                if key in cache:
                    cached_val: EmbeddingRecord = cache[key]  # type: ignore[assignment]
                    hashed[key]["embedding"] = cached_val["embedding"]

    # Identify texts that still need to be embedded.
    missing_keys = [k for k, v in hashed.items() if v["embedding"] is None]
    if missing_keys:
        texts_to_embed = [hashed[k]["text"] for k in missing_keys]
        new_embs = run_embeddings(texts_to_embed, base_url=base_url, model=model)
        for k, emb in zip(missing_keys, new_embs):
            hashed[k]["embedding"] = emb
            if cache is not None:
                # Safe because value matches *EmbeddingRecord* schema.
                cache[k] = hashed[k]  # type: ignore[assignment]

    # Return embeddings in the same order as *text_list* (handles duplicates).
    return [hashed[hashlib.sha256(t.encode()).hexdigest()]["embedding"] for t in text_list]

# embeddings = run_embeddings_with_cache(sample_texts,cache_path = expanduser("~/data/embedding_cache/"))
