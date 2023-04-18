from typing import List

import openai_api
from utils import newline, shorten
from caching import Caching, OpenAICacheKey, OpenAICacheValue


class OpenAIWrapper:
        def __init__(self, cache_path:str=None, save_every_n_seconds: int=600):
            self.cache = Caching(cache_path=cache_path, save_every_n_seconds=save_every_n_seconds)

        def call(self, prompt, engine, max_tokens=300, stop_token="###", temperature=0.0):
            if not prompt:
                return ""
            cache_key = OpenAICacheKey(engine=engine,
                                       prompt=prompt.lstrip(), # don't store new lines in the beginning.
                                       stop_token=stop_token,
                                       temperature=temperature,
                                       max_tokens=max_tokens)
            cache_val = self.cache.get(key=cache_key)
            if not cache_val:
                print(f"\nCalling GPT3: {shorten(prompt, max_words=10)}...")
                val_dict = openai_api.OpenaiAPIWrapper.call(prompt=prompt,
                                                 engine=engine,
                                                 max_tokens=max_tokens,
                                                 stop_token=stop_token,
                                                 temperature=temperature)
                cache_val = self.cache.set(key=cache_key, value=OpenAICacheValue(
                    first_response=str(openai_api.OpenaiAPIWrapper.get_first_response(response=val_dict, engine=engine))
                ))

            return cache_val.first_response

        def mk_cache_key(self, prompt: str, engine: str, max_tokens=300, stop_token="###", temperature=0.0) -> str:
            cache_key = OpenAICacheKey(engine=engine,
                                  prompt=prompt.lstrip(), # don't store new lines in the beginning.
                                  stop_token=stop_token,
                                  temperature=temperature,
                                  max_tokens=max_tokens)
            return cache_key


        def call_batch(self, prompts: List[str], engine: str, max_tokens=300, stop_token="###", temperature=0.0):
            cache_keys = [self.mk_cache_key(prompt=prompt, engine=engine, stop_token=stop_token, temperature=temperature, max_tokens=max_tokens)  for prompt in prompts]
            precached_entries = {prompt: self.cache.get(key=cache_key) for cache_key, prompt in zip(cache_keys, prompts)}
            uncached_prompts   = [prompt for prompt, cached_entry in precached_entries.items() if not cached_entry]

            print(f"\nCalling GPT3 as a batch for ({len(uncached_prompts)}/ {len(precached_entries)}) "
                  f"prompts: {(newline+newline).join([shorten(p, max_words=10)+ '...' for p in uncached_prompts])}")

            batched_response = openai_api.OpenaiAPIWrapper.call(prompt=uncached_prompts,
                                                        engine=engine,
                                                        max_tokens=max_tokens, stop_token=stop_token,
                                                        temperature=temperature)

            for prompt, completion in zip(uncached_prompts, openai_api.OpenaiAPIWrapper.get_first_response_batched(response=batched_response, engine=engine)):
                value = OpenAICacheValue(first_response= completion.strip())
                precached_entries[prompt] = value
                cache_key = self.mk_cache_key(prompt=prompt, engine=engine, stop_token=stop_token, temperature=temperature, max_tokens=max_tokens)
                self.cache.set(key=cache_key, value=value)

            return [p.first_response for _, p in precached_entries.items()]

