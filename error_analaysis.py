import argparse
import ast
import json
import os.path
from pathlib import Path

from gptinference import utils
from gptinference.utils import read_jsonl_or_json, write_json

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
        print("================================= user_id: {}, topic: {} =================================".format(user_id, topic))
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
    print(sum(list(ctr_dict.values())), total)

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
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)