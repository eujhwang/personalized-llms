import argparse
import ast
import json
import math
import os.path
import random
from pathlib import Path

import pandas as pd
import tqdm
from gptinference import utils
from gptinference.utils import read_jsonl_or_json, write_json
from sklearn.metrics import cohen_kappa_score

from synthesize_opinionqa.utils import set_seed

pd.options.display.max_columns = 10
# pd.options.display.max_columns = 10


SURVEY_TO_TOPIC = {
    'American_Trends_Panel_W26': "Guns",
    'American_Trends_Panel_W27': "Automation and driverless vehicles",
    'American_Trends_Panel_W29': "Views on gender",
    'American_Trends_Panel_W32': "Community types & sexual harassment",
    'American_Trends_Panel_W34': "Biomedical & food issues",
    'American_Trends_Panel_W36': "Gender & Leadership",
    'American_Trends_Panel_W41': "America in 2050",
    'American_Trends_Panel_W42': "Trust in science",
    'American_Trends_Panel_W43': "Race",
    'American_Trends_Panel_W45': "Misinformation",
    'American_Trends_Panel_W49': "Privacy & Surveilance",
    'American_Trends_Panel_W50': "Family & Relationships",
    'American_Trends_Panel_W54': "Economic inequality",
    'American_Trends_Panel_W82': "Global attitudes",
    'American_Trends_Panel_W92': "Political views"
}
OUTPUT_MAP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def collapse_class(choices, model_choice, user_choice):
    total_len = len(choices) -1
    # if total_len % 2 == 0: # even number -> mid=2
    mid = total_len // 2

    alphabet_choices = [OUTPUT_MAP[i] for i in range(len(choices))]
    if model_choice in alphabet_choices[:mid] and user_choice in alphabet_choices[:mid]:
        # print("corret1. model_choice:", model_choice, "user_choice:", user_choice, "choices:", choices, "alphabet_choices:", alphabet_choices)
        return True
    elif model_choice in alphabet_choices[mid:] and user_choice in alphabet_choices[mid:]:
        # print("corret2. model_choice:", model_choice, "user_choice:", user_choice, "choices:", choices, "alphabet_choices:", alphabet_choices)
        return True
    else:
        # print("incorrect. model_choice:", model_choice, "user_choice:", user_choice, "choices:", choices, "alphabet_choices:", alphabet_choices)
        return False


def collapsed_class_accuracy(gen_out, qinfo_dict):
    """
    collapse classes into 2 classes and calculate accuracy
    """
    accuracy_list = []
    user_accuracy_list = []
    for i, (gen) in enumerate(gen_out):
        user_id = gen['user_id']
        topic = gen['topic']
        gen_out_list = gen['generated_output']
        # print("================================= user_id: {}, topic: {} =================================".format(user_id, topic))
        correct, incorrect = 0, 0
        for gen_out in gen_out_list:
            qid = gen_out['qid']
            user_choice = gen_out['user_choice']
            model_choice = gen_out['model_choice']

            if len(user_choice) > 1 or len(model_choice) > 1:
                continue

            choices = [choice for choice in ast.literal_eval(qinfo_dict[qid]["choice"])]
            user_answer = choices[OUTPUT_MAP.index(user_choice)]
            model_answer = choices[OUTPUT_MAP.index(model_choice)]

            if user_answer.lower() == "refused" and model_answer.lower() == "refused":
                continue

            is_correct = collapse_class(choices, model_choice, user_choice)
            if is_correct:
                correct += 1
            if not is_correct:
                incorrect += 1
        total_num = correct + incorrect
        accuracy_per_user = correct / total_num
        user_accuracy_list.append({user_id: accuracy_per_user})
        accuracy_list.append(accuracy_per_user)

    final_accuracy = sum(accuracy_list) / len(accuracy_list)
    return {"accuracy": final_accuracy, "user-accuracy": user_accuracy_list}


def count_predictions(imp_gen, exp_gen, imexp_gen, qinfo_dict):
    """
    for venn diagram
    """
    assert len(imp_gen) == len(exp_gen) == len(imexp_gen)

    ctr_dict = {
        "case1": 0, "case2": 0, "case3": 0, "case4": 0,
        "case5": 0, "case6": 0, "case7": 0, "case8": 0,
    }
    total = 0
    for i, (imp, exp, imexp) in enumerate(zip(imp_gen, exp_gen, imexp_gen)):
        assert imp["user_id"] == exp["user_id"] == imexp["user_id"]

        # case1: imp:x, exp: x, imexp: x
        # case2: imp:o, exp: x, imexp: x
        # case3: imp:x, exp: o, imexp: x
        # case4: imp:x, exp: x, imexp: o
        # case5: imp:o, exp: o, imexp: x
        # case6: imp:o, exp: x, imexp: o
        # case7: imp:x, exp: o, imexp: o
        # case8: imp:o, exp: o, imexp: o

        imp_out_list = imp['generated_output']
        exp_out_list = exp['generated_output']
        imexp_out_list = imexp['generated_output']
        for imp_out, exp_out, imexp_out in zip(imp_out_list, exp_out_list, imexp_out_list):
            assert imp_out['qid'] == exp_out['qid'] == imexp_out['qid']
            qid = imp_out['qid']
            user_choice = imp_out['user_choice']

            if len(user_choice) > 1:
                continue

            choices = [choice for choice in ast.literal_eval(qinfo_dict[qid]["choice"])]
            user_answer = choices[OUTPUT_MAP.index(user_choice)]

            if user_answer.lower() == "refused":
                continue

            imp_model_choice = imp_out['model_choice']
            exp_model_choice = exp_out['model_choice']
            imexp_model_choice = imexp_out['model_choice']

            if imp_model_choice != user_choice and exp_model_choice != user_choice and imexp_model_choice != user_choice:
                ctr_dict["case1"] += 1
            elif imp_model_choice == user_choice and exp_model_choice != user_choice and imexp_model_choice != user_choice:
                ctr_dict["case2"] += 1
            elif imp_model_choice != user_choice and exp_model_choice == user_choice and imexp_model_choice == user_choice:
                ctr_dict["case3"] += 1
            elif imp_model_choice != user_choice and exp_model_choice != user_choice and imexp_model_choice == user_choice:
                ctr_dict["case4"] += 1
            elif imp_model_choice == user_choice and exp_model_choice == user_choice and imexp_model_choice != user_choice:
                ctr_dict["case5"] += 1
            elif imp_model_choice == user_choice and exp_model_choice != user_choice and imexp_model_choice == user_choice:
                ctr_dict["case6"] += 1
            elif imp_model_choice != user_choice and exp_model_choice == user_choice and imexp_model_choice == user_choice:
                ctr_dict["case7"] += 1
            else: # imp_model_choice == user_choice and exp_model_choice == user_choice and imexp_model_choice == user_choice:
                ctr_dict["case8"] += 1
            total += 1
    print("ctr_dict:", ctr_dict)
    assert sum(list(ctr_dict.values())) == total
    print(sum(list(ctr_dict.values())), total)


def get_user_demographic(user_responses):
    user_to_demo = {}
    for user_resp in user_responses:
        user_id = user_resp["user_id"]
        explicit_persona = user_resp["explicit_persona"]
        user_to_demo[user_id] = explicit_persona
    return user_to_demo

def get_user_topic(user_responses):
    user_to_topic = {}
    for user_resp in user_responses:
        user_id = user_resp["user_id"]
        topic = user_resp["topic"]
        user_to_topic[user_id] = topic
    return user_to_topic

def get_user_declarative(user_responses, num_implicit=16):
    user_to_decl = {}
    for user_resp in user_responses:
        user_id = user_resp["user_id"]
        implicit_persona = user_resp["implicit_persona"]
        implicit_persona = utils.take(arr=implicit_persona, num=num_implicit)
        user_to_decl[user_id] = [item["declarative_opinion"] for item in implicit_persona]
    return user_to_decl

def extract_error_samples(imp_gen, exp_gen, imexp_gen, qinfo_dict, user_responses):
    assert len(imp_gen) == len(exp_gen)

    user_to_demo = get_user_demographic(user_responses)
    user_to_topic = get_user_topic(user_responses)
    user_to_decl = get_user_declarative(user_responses, num_implicit=16)

    # case1: imp:x, exp: o
    error_list = []
    for i, (imp, exp, imexp) in enumerate(zip(imp_gen, exp_gen, imexp_gen)):
        assert imp["user_id"] == exp["user_id"] == imexp["user_id"]

        user_id = imp["user_id"]
        imp_out_list = imp['generated_output']
        exp_out_list = exp['generated_output']
        imexp_out_list = imexp['generated_output']

        for imp_out, exp_out, imexp_out in zip(imp_out_list, exp_out_list, imexp_out_list):
            assert imp_out['qid'] == exp_out['qid'] == imexp_out['qid']
            qid = imp_out['qid']
            user_choice = imp_out['user_choice']
            imp_model_choice = imp_out['model_choice']
            exp_model_choice = exp_out['model_choice']
            imexp_model_choice = imexp_out['model_choice']

            if len(user_choice) > 1 or len(imp_model_choice) > 1 or len(exp_model_choice) > 1 or len(imexp_model_choice) > 1:
                continue

            choices = [choice for choice in ast.literal_eval(qinfo_dict[qid]["choice"])]
            user_answer = choices[OUTPUT_MAP.index(user_choice)]
            if user_answer.lower() == "refused":
                continue

            imp_model_answer = choices[OUTPUT_MAP.index(imp_model_choice)]
            exp_model_answer = choices[OUTPUT_MAP.index(exp_model_choice)]
            imexp_model_answer = choices[OUTPUT_MAP.index(imexp_model_choice)]

            if imp_model_choice != user_choice and exp_model_choice == user_choice:
                error_list.append({
                    "user_id": user_id,
                    "qid": qid,
                    "topic": user_to_topic[user_id],
                    "question": qinfo_dict[qid]['question'],
                    "choice": choices,
                    "opinions": user_to_decl[user_id],
                    "demographic": user_to_demo[user_id],
                    "user_answer": user_answer,
                    "imp_model_answer": imp_model_answer,
                    "exp_model_answer": exp_model_answer,
                    "imexp_model_answer": imexp_model_answer,
                })
    # print("error_list:", len(error_list), error_list[0])
    sampled_errors = random.sample(error_list, 50)
    df = pd.DataFrame.from_dict(sampled_errors)
    print("sampled_errors:", len(sampled_errors), sampled_errors[0])
    print("df:", len(df), df)
    df.to_csv("sampled_error.csv")

def save_metrics(file_path, metrics):
    # Path(file_path).mkdir(parents=True, exist_ok=True)
    write_json(outpath=file_path, json_data=metrics)

def load_resources(dirs, type: str = "gen"):
    gen_file = "model_generation.json"
    accu_file = "model_accuracy.json"

    if type == "gen":
        no_per_gen = read_jsonl_or_json(os.path.join(dirs["no_per_dir"], gen_file))
        imp_gen = read_jsonl_or_json(os.path.join(dirs["imp_dir"], gen_file))
        exp_gen = read_jsonl_or_json(os.path.join(dirs["exp_dir"], gen_file))
        imexp_gen = read_jsonl_or_json(os.path.join(dirs["imexp_dir"], gen_file))
        return no_per_gen, imp_gen, exp_gen, imexp_gen

    if type == "accu":
        no_per_accu = read_jsonl_or_json(os.path.join(dirs["no_per_dir"], accu_file))
        imp_accu = read_jsonl_or_json(os.path.join(dirs["imp_dir"], accu_file))
        exp_accu = read_jsonl_or_json(os.path.join(dirs["exp_dir"], accu_file))
        imexp_accu = read_jsonl_or_json(os.path.join(dirs["imexp_dir"], accu_file))
        return no_per_accu, imp_accu, exp_accu, imexp_accu

    return None, None, None, None


def check_same_demographic(user1_demo, user2_demo):
    for key in user1_demo.keys():
        user1_value = user1_demo[key]
        if key not in user2_demo:
            continue

        user2_value = user2_demo[key]
        if user1_value != user2_value:
            return False
    return True

def check_similar_opinion(user1_op, user2_op):
    user1_qid_set = user1_op.keys()
    user2_qid_set = user2_op.keys()
    common_qid_set = user1_qid_set & user2_qid_set

    same_op = 0
    for qid in list(common_qid_set):
        user1_answer = user1_op[qid]['answer']
        user2_answer = user2_op[qid]['answer']
        if user1_answer == user2_answer:
            same_op += 1

    return same_op

def get_user_op(user1_op, user2_op):
    user1_qid_set = user1_op.keys()
    user2_qid_set = user2_op.keys()
    common_qid_set = user1_qid_set & user2_qid_set

    user1_answer_list, user2_answer_list = [], []
    for qid in list(common_qid_set):
        user1_answer = user1_op[qid]['answer']
        user2_answer = user2_op[qid]['answer']
        user1_answer_list.append(user1_answer)
        user2_answer_list.append(user2_answer)
    return user1_answer_list, user2_answer_list


def calculate_demo_match_percent(user1_demo, user2_demo):
    user1_demo_set = user1_demo.keys()
    user2_demo_set = user2_demo.keys()
    common_demo_set = list(user1_demo_set & user2_demo_set)

    same_demo = 0
    invalid = 0
    for demo_name in common_demo_set:
        user1_demo_val = user1_demo[demo_name]
        user2_demo_val = user2_demo[demo_name]

        if (isinstance(user1_demo_val, float) and math.isnan(user1_demo_val)) or (isinstance(user2_demo_val, float) and math.isnan(user2_demo_val)):
            invalid += 1
            continue
        if user1_demo[demo_name].lower() == "refused" or user2_demo[demo_name].lower() == "refused":
            invalid += 1
            continue

        if user1_demo[demo_name] == user2_demo[demo_name]:
            same_demo += 1
    demo_similarity = same_demo / (len(common_demo_set)-invalid)
    return demo_similarity


def same_demo_diff_op(all_user_responses):
    print("same_demo_diff_op:\n", all_user_responses[0].keys())
    total_num = len(all_user_responses)

    survey_to_agree_ratio = {}
    all_agree_ratio = []
    for i in tqdm.tqdm(range(total_num)):
        user1_id = all_user_responses[i]["user_id"]
        user1_survey = all_user_responses[i]["survey_name"]
        user1_demo = all_user_responses[i]["explicit_info"]
        user1_op = all_user_responses[i]["implicit_info"]
        for j in range(i, total_num):
            user2_id = all_user_responses[j]["user_id"]
            user2_survey = all_user_responses[j]["survey_name"]
            user2_demo = all_user_responses[j]["explicit_info"]
            user2_op = all_user_responses[j]["implicit_info"]
            if i == j or user1_id == user2_id or user1_survey != user2_survey :
                continue

            is_same_demo = check_same_demographic(user1_demo, user2_demo)
            if is_same_demo:
                user1_resp_list, user2_resp_list = get_user_op(user1_op, user2_op)
                agreement_score = cohen_kappa_score(user1_resp_list, user2_resp_list)

                if user1_survey not in survey_to_agree_ratio:
                    survey_to_agree_ratio[user1_survey] = []

                survey_to_agree_ratio[user1_survey].append(agreement_score)
                all_agree_ratio.append(agreement_score)

    _survey_to_agree_ratio = {}
    for survey, score_list in survey_to_agree_ratio.items():
        _survey_to_agree_ratio[survey] = sum(survey_to_agree_ratio[survey]) / len(survey_to_agree_ratio[survey])

    with open("same_demo_diff_op_score.json", "w") as f:
        json.dump({
            "final_score": sum(all_agree_ratio)/len(all_agree_ratio),
            "score_per_survey": _survey_to_agree_ratio
        }, f, indent=4)
    print("survey_to_agree_ratio:", _survey_to_agree_ratio)
    print("final_agree_ratio:", sum(all_agree_ratio)/len(all_agree_ratio))


def same_op_diff_demo(all_user_responses, num_implicit=4):
    print("same_op_diff_demo:\n", all_user_responses[0].keys())
    total_num = len(all_user_responses)

    op_to_demo_score = {}
    for i in tqdm.tqdm(range(total_num)):
        user1_id = all_user_responses[i]["user_id"]
        user1_survey = all_user_responses[i]["survey_name"]
        user1_demo = all_user_responses[i]["explicit_info"]
        user1_op = all_user_responses[i]["implicit_info"]
        for j in range(i, total_num):
            user2_id = all_user_responses[j]["user_id"]
            user2_survey = all_user_responses[j]["survey_name"]
            user2_demo = all_user_responses[j]["explicit_info"]
            user2_op = all_user_responses[j]["implicit_info"]
            if i == j or user1_id == user2_id or user1_survey != user2_survey:
                continue

            num_same_op = check_similar_opinion(user1_op, user2_op)
            demo_similarity = calculate_demo_match_percent(user1_demo, user2_demo)

            if num_same_op not in op_to_demo_score:
                op_to_demo_score[num_same_op] = []
            op_to_demo_score[num_same_op].append(demo_similarity)

    all_op_to_demo_score = {}
    for num_op, demo_score_list in op_to_demo_score.items():
        all_op_to_demo_score[num_op] = sum(op_to_demo_score[num_op]) / len(op_to_demo_score[num_op])
    # print("all_op_to_demo_score:", all_op_to_demo_score)

    # aggregate scores based on num_implicit
    score_list = []
    for num_op, demo_score in all_op_to_demo_score.items():
        if num_op >= num_implicit:
            score_list.append(demo_score)
    print(f"final score base on the samples with more than {num_implicit} opinions: {sum(score_list)/len(score_list)}")
    with open("same_op_diff_demo_score.json", "w") as f:
        json.dump({
            "final_score": sum(score_list)/len(score_list),
            "score_per_op": all_op_to_demo_score
        }, f, indent=4)


def main(args):
    root_dir = "data"
    dirs = {
        "no_per_dir": os.path.join(root_dir, "model-output", "no-persona-t-1-u35-q30"),
        "imp_dir": os.path.join(root_dir, "model-output", "implicit_16pts-t-1-u35-q30"),
        "exp_dir": os.path.join(root_dir, "model-output", "explicit-t-1-u35-q30"),
        "imexp_dir": os.path.join(root_dir, "model-output", "imp-16pts_exp-t-1-u35-q30"),
    }

    no_per_gen, imp_gen, exp_gen, imexp_gen = load_resources(type="gen", dirs=dirs)
    no_per_accu, imp_accu, exp_accu, imexp_accu = load_resources(type="accu", dirs=dirs)

    qinfo_dict = read_jsonl_or_json(os.path.join(root_dir, "opinionqa", "all_qinfo_dict.json"))
    user_responses = read_jsonl_or_json(os.path.join(root_dir, "opinionqa", "sampled_user_responses_decl.json"))

    count_predictions(imp_gen, exp_gen, imexp_gen, qinfo_dict)
    no_per_collapsed_accu = collapsed_class_accuracy(no_per_gen, qinfo_dict)
    imp_collapsed_accu = collapsed_class_accuracy(imp_gen, qinfo_dict)
    exp_collapsed_accu = collapsed_class_accuracy(exp_gen, qinfo_dict)
    imexp_collapsed_accu = collapsed_class_accuracy(imexp_gen, qinfo_dict)

    save_metrics(os.path.join(dirs["no_per_dir"], "clpse_accuracy.json"), no_per_collapsed_accu)
    save_metrics(os.path.join(dirs["imp_dir"], "clpse_accuracy.json"), imp_collapsed_accu)
    save_metrics(os.path.join(dirs["exp_dir"], "clpse_accuracy.json"), exp_collapsed_accu)
    save_metrics(os.path.join(dirs["imexp_dir"], "clpse_accuracy.json"), imexp_collapsed_accu)

    # correct answer with explicit info, but incorrect answer with implicit info
    # extract_error_samples(imp_gen, exp_gen, imexp_gen, qinfo_dict, user_responses)

    all_user_responses = read_jsonl_or_json("data/all_user_responses.json")
    # sample_path = "sampled1000_user_responses.json"
    # sampled_responses = random.sample(sampled_user_responses, 1000)
    # with open(sample_path, "w") as f:
    #     json.dump(sampled_responses, f)

    # sampled_user_responses = read_jsonl_or_json("sampled1000_user_responses.json")
    same_demo_diff_op(all_user_responses)
    same_op_diff_demo(all_user_responses, num_implicit=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    set_seed(42)
    main(args)