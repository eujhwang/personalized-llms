import json

from gptinference import utils
from tqdm import tqdm

if __name__ == '__main__':
    in_path  = "data/opinionqa/sampled_user_responses_20_decl_topk.json"
    # in_path  = "data/opinionqa/small_sample_user_resp.json"
    out_path = "data/opinionqa/sampled_user_responses_20_decl_topk_compressed.jsonl"

    print(f"Reading input from {in_path}")

    with open(out_path, 'w') as outfile:
        for j in tqdm(utils.read_jsonl_or_json(in_path)):
            # Remove "emb"
            # j["emb"] = []
            # Remove "implicit_persona" "emb"
            for x in j["implicit_persona"]:
                x["emb"]=[]
            for x in j["implicit_questions"]:
                x["emb"]=[]

            json.dump(j, outfile)
            outfile.write('\n')

    print(f"Output is in {out_path}")

