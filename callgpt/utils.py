import json
import os
from typing import List

newline = "\n"

def take(num: int, arr: List):
    if num ==0:
        return []
    if num == -1:
        return arr
    if num < 0:
        return arr
    num = min(len(arr), num)
    return arr[: num]

def write_jsonl(outpath, data_points, append=False) -> bool:
    with open(outpath, 'w' if not append else 'a') as open_file:
        for item in data_points:
            json.dump(item, open_file)
            open_file.write('\n')
        return True

def write_json(outpath, json_data, indent=4, append=False) -> bool:
    with open(outpath, 'w' if not append else 'a') as open_file:
        json.dump(json_data, open_file, indent=indent)
    return True

def read_jsonl_or_json(path):
    if not os.path.exists(path):
        raise Exception('File expected at ' + path + ' not found')
    records = []
    with open(path)  as f:
        if path.endswith('.json'):
            records = json.load(f)
        elif path.endswith('.jsonl'):
            records = [json.loads(l) for l in f]
    return records

def write_txt(outpath, data_str, append=False) -> bool:
    with open(outpath, 'w' if not append else 'a') as open_file:
        open_file.write(data_str)
    return True

def shorten(s: str, max_words: int):
    if not s:
        return ""
    words = s.split(" ")
    clipped_txt = " ".join(words[: min(len(words), int(max_words))])
    return f"{clipped_txt}{'...' if clipped_txt!=s else ''}"

def only_a_z(s: str) -> str:
    """
    Lowercases and strips anything other a-z
    :param s:
    :return:
    """
    if not s:
        return ""
    chars = [ch if 'a' <= ch <= 'z' else '' for ch in s.lower()]
    return ''.join([ch for ch in chars if ch])