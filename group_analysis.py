import ast
import json

from gptinference.utils import read_jsonl_or_json
import tqdm
import math
from collections import Counter

def main():
    # all_user_responses = read_jsonl_or_json("data/all_user_responses.json")
    qinfo_dict = read_jsonl_or_json("data/opinionqa/all_qinfo_dict.json")
    user_responses = read_jsonl_or_json("data/opinionqa/sampled_user_responses_20.json")
    qid_to_ans = {}
    for user_resp in user_responses:
        user_id = user_resp["user_id"]
        topic = user_resp["topic"]
        implicit_persona = user_resp["implicit_persona"]
        explicit_persona = user_resp["explicit_persona"]
        implicit_questions = user_resp["implicit_questions"]

        for exp_per in explicit_persona:
            if "Political party" not in exp_per:
                continue
            polparty = exp_per["Political party"]

        if polparty == "Refused" or (isinstance(polparty, float) and math.isnan(polparty)):
            continue

        for imp_que in implicit_questions:
            qid = imp_que["qid"]
            ans = imp_que["answer"]

            if qid not in qid_to_ans:
                qid_to_ans[qid] = {"majority": []}
            if polparty not in qid_to_ans[qid]:
                qid_to_ans[qid][polparty] = []
            qid_to_ans[qid]["majority"].append(ans)
            qid_to_ans[qid][polparty].append(ans)


    major_qid_to_ans = {}
    for qid in qid_to_ans.keys():
        for group, resps in qid_to_ans[qid].items():
            counter = Counter(resps)
            total = sum(counter.values())
            most_common_ans = counter.most_common(1)[0]
            proportion = most_common_ans[-1]/total
            if proportion > 0.5:
                if qid not in major_qid_to_ans:
                    major_qid_to_ans[qid] = {}

                major_qid_to_ans[qid]["question"] = qinfo_dict[qid]["question"]
                choice = ast.literal_eval(qinfo_dict[qid]["choice"])
                if choice[-1].lower() == "refused":
                    choice = choice[:-1]
                major_qid_to_ans[qid]["choice"] = choice
                major_qid_to_ans[qid][group] = (most_common_ans[0], proportion)
    print("major_qid_to_ans: {}".format(major_qid_to_ans))

    with open(f"analysis/group_level_test.json", "w") as f:
        json.dump(major_qid_to_ans, f, indent=4)

if __name__ == '__main__':
    main()