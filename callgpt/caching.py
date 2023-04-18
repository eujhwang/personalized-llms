import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

from utils import read_jsonl_or_json, write_txt, write_jsonl


import atexit

DEFAULT_MAX_TOKENS = 300

@dataclass
class OpenAICacheKey:
    engine: str
    prompt: str
    stop_token: str
    temperature: float
    max_tokens: float = field(default=DEFAULT_MAX_TOKENS)

    def __eq__(self, other):
        return other and \
               self.engine == other.engine and self.prompt == other.prompt and \
               self.stop_token == other.stop_token and self.temperature == other.temperature and \
               self.max_tokens == other.max_tokens

    def __hash__(self):
        return hash((self.engine, self.prompt, self.stop_token, self.temperature, self.max_tokens))

    def to_json(self):
        j = {"engine": self.engine,
                "prompt": self.prompt,
                "stop_token": self.stop_token,
                "temperature": self.temperature,
                }
        if self.max_tokens != DEFAULT_MAX_TOKENS:
            j["max_tokens"] = self.max_tokens
        return j

    @staticmethod
    def from_json(d):
        k = OpenAICacheKey(
            engine=d["engine"],
            prompt=d["prompt"],
            stop_token=d["stop_token"],
            temperature=float(d["temperature"]),
        )
        if "max_tokens" in d:
            k.max_tokens = float(d["max_tokens"])
        return k



@dataclass
class OpenAICacheValue:
    first_response: str
    def __eq__(self, other):
        return other and \
               self.first_response == other.first_response

    def __hash__(self):
        return hash(self.first_response)

    def to_json(self):
        return {"first_response": self.first_response}

    @staticmethod
    def from_json(d):
        return OpenAICacheValue(
            first_response=d["first_response"]
        )

class Caching:

    def __init__(self, cache_path: str, save_every_n_seconds = 600):

        # this function is called twice:
        # https://stackoverflow.com/questions/50142921/prevent-method-from-being-called-twice-in-a-class-that-can-be-called-with-with
        atexit.register(self.cleanup)

        self.save_every_n_seconds = save_every_n_seconds
        self.cache_started_at = time.time()
        self.cache: Dict[OpenAICacheKey, OpenAICacheValue] = {}
        assert cache_path is not None, f"\n\ncaching openai: cache file is empty."
        self.cache_path: str = cache_path.strip()
        print(f"\nCaching: Loading cache from {self.cache_path}", end=" ... ")
        cache_content_list = self.load_cache(self.cache_path)
        # json.loads() is used twice to convert from serialized dict in str format to dict
        if cache_content_list:
            for x in cache_content_list:
                j = json.loads(x)
                k = OpenAICacheKey.from_json(j)
                k.prompt = k.prompt.lstrip()  # strip leading whitespaces.
                v = OpenAICacheValue.from_json(j)
                self.cache[k] = v
        print(f"[done]")

    def cleanup(self):
        # works similar to a destructor but does not offload builtins.open method.
        print(f"\nFinal cleanup cache saving...", end="...")
        self.save_cache()

    def load_cache(self, fp) -> List:
        if not os.path.exists(fp) or os.path.getsize(fp) == 0:
            write_txt(outpath=fp, data_str="")
            return []
        else:
            return read_jsonl_or_json(fp)

    def get(self, key: OpenAICacheKey) -> OpenAICacheValue:
        assert key is not None and key, f"caching: cache key is empty ({key})."
        return self.cache.get(key, None)

    def set(self, key, value) -> OpenAICacheValue:
        secs_since_last_save = time.time() - self.cache_started_at
        if secs_since_last_save % self.save_every_n_seconds == 0:
            print(f"\nPeriodic cache saving... with {len(self.cache.items())} pts ... ", end=" ... ")
            self.save_cache()
        assert key is not None and key, f"caching: cache key is wrong ({key})."
        assert value is not None and len(f"{value}")>0, f"caching: value to cache is empty ({value})."
        # print(f"\nCache addition happening: before {len(self.cache.items())} pts ", end=" ... ")
        self.cache[key] = value
        # print(f" after {len(self.cache.items())} pts ")
        return value

    def save_cache(self):
        print(f"\nCaching: Saving openai cache [{self.cache_path}] with {len(self.cache.items())} pts in ", end=" ... ")
        data_points = []
        for x, y in self.cache.items():
            key_with_value = x.to_json()
            key_with_value["first_response"] = y.first_response
            data_points.append(json.dumps(key_with_value))
        print(f" {len(data_points)} lines", end=" .")
        write_jsonl(data_points=data_points, outpath=self.cache_path, append=False)
        print("done.")
