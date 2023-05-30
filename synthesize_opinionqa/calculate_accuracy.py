import argparse
import os
from os.path import isfile, join, isdir
from pathlib import Path
from typing import List

import pandas as pd
from gptinference import utils
from gptinference.utils import read_jsonl_or_json, write_json

from personalized_opinionqa.mean_responses_qa import collapsed_class


def calculate_topicwise_accuracy(in_path, metrics_files: List[str]):
    user_responses = read_jsonl_or_json(in_path)
    user_id_to_topic = {}
    for resp in user_responses:
        user_id = resp["user_id"]
        topic = resp["topic"]
        user_id_to_topic[str(user_id)] = topic

    dir_to_topic_accu = {}
    for metrics_file in metrics_files:
        accu_results = read_jsonl_or_json(metrics_file)
        user_accuracy_list = accu_results["user-accuracy"]
        topic_to_accu_list = {}
        for user_accuracy in user_accuracy_list:
            for user_id, accuracy in user_accuracy.items():
                topic = user_id_to_topic[user_id]

                if topic not in topic_to_accu_list.keys():
                    topic_to_accu_list[topic] = []

                topic_to_accu_list[topic].append(accuracy)

        topic_to_accu = []
        for topic, accu_list in topic_to_accu_list.items():
            accuracy = sum(accu_list) / len(accu_list)
            topic_to_accu.append({"topic": topic, "accuracy": accuracy})

        basedir = Path(metrics_file).parts[-2]
        dir_to_topic_accu[basedir] = topic_to_accu
    return dir_to_topic_accu

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--in_path", type=str, default="data/opinionqa/sampled_user_responses_decl.json", help="synthesized opinionqa json path")
    parser.add_argument("--in_path", type=str, default="data/opinionqa/sampled_user_responses_20_decl_topk_compressed.jsonl", help="synthesized opinionqa json path")
    parser.add_argument("--out_path", type=str, default="data/model-output-topk/topicwise_accuracy.json", help="output metrics json path")
    parser.add_argument("--model_outputs_path", type=str, default="data/model-output-topk", help="model output directory json path")
    args = parser.parse_args()

    # add topic-wise accuracy when there are multiple dirs
    model_output_files_randomk = {
                          "no-persona-t-1-u35-q30": "no-persona",
                          "explicit-t-1-u35-q30": "explicit",
                          # "imp-8pts_exp-t-1-u35-q30": "explicit+imp8",
                          "imp-16pts_exp-t-1-u35-q30": "explicit+imp16",
                          # "implicit_8pts-t-1-u35-q30": "implicit8",
                          # "implicit_16pts-t-1-u35-q30": "implicit16",
                         }
    model_output_files_topk = {
        "imp-8pts_exp-t-1-u35-q30": "explicit+imp8+topk"
    }

    model_output_files = model_output_files_topk
    metrics_files = [os.path.join(args.model_outputs_path, f"{x}/model_accuracy.json") for x in model_output_files.keys()]

    topic_metrics = calculate_topicwise_accuracy(args.in_path, metrics_files)

    # print(pd.DataFrame(topic_metrics))

    topics = []
    tbl = []
    topic_header = []
    model_header = []
    for model_name, arr_of_topic_accuracy_dict in topic_metrics.items():
        if not topics:
            topics = [x["topic"] for x in arr_of_topic_accuracy_dict]
            topic_header = [x.replace("&", ",") for x in topics]  # header

        ta_arr = []
        for topic in topics:
            for ta in arr_of_topic_accuracy_dict:
                if ta["topic"] == topic:
                    ta_arr.append(float(ta["accuracy"]))
        model_header.append(model_output_files[model_name])
        tbl.append(ta_arr)

    write_json(outpath=args.out_path, json_data=topic_metrics)
    print(f"Topicwise accuracy output in {args.out_path}")
    df = pd.DataFrame(tbl)
    df.columns = topic_header

    df = df.transpose().round(2)
    df.columns = model_header
    print(df)
    print(f"\n\nLatex table for paper:\n")
    print(df.to_latex(float_format="{:.2f}".format))


from math import sqrt

def wilson(p, n, z = 2.58): #1.96):
    # usage: wilson(float(frac_correct)/100.0, tot)
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z*z / (2*n)
    adjusted_standard_deviation = sqrt((p*(1 - p) + z*z / (4*n)) / n)

    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    return lower_bound, upper_bound, f"$\pm$ {100.0*(p - lower_bound) :0.2f}"


def compute_collapsed_accuracy(predictions_evaluated_json_fp, user_responses_json_fp):
    """
    @param:
        user_responses_json_fp: data/opinionqa/sampled_user_responses_20_decl_topk_compressed.jsonl
        "implicit_questions": [
            {
                "qid": "GUNRESPKIDSB_W26",
                "question": "Thinking about gun owners who have ...",
                "choices": [
                    "Essential",
                    "Important but not essential",
                    "Not important",
                    "Should not be done",
                    "Refused"
                ],
                "answer": "Essential",
                "subtopic_cg": [
                    "crime/security"
                ],


        predictions_evaluated_json_fp: data/model-output-topk/imp-8pts_exp-t-1-u35-q30/model_generation.json
        [
            {
                "user_id": 53,
                "topic": "Guns",
                "generated_output": [
                    {
                        "model_choice": "A",
                        "user_choice": "A",
                        "qid": "GUNCONTRIBB_W26"
                    },
                ....
            }
        ]
    """
    users_dict = dict()
    for user_details in utils.read_jsonl_or_json(user_responses_json_fp):
        # for each user id as key, store qid -> [choices] arr.
        dd = {ques_detail["qid"]: [x for x in ques_detail["choices"] if x!="Refused"] for ques_detail in user_details["implicit_questions"]}
        users_dict[user_details["user_id"]] = dd

    choices_dict = {chr(i): (i-65) for i in range(65, 91)} # {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    evals = []
    evals_collapsed = []
    for user_questions in utils.read_jsonl_or_json(predictions_evaluated_json_fp):
        user_id = user_questions["user_id"]
        for uq in user_questions["generated_output"]:
            is_correct = uq["model_choice"] == uq["user_choice"]
            choices = users_dict[user_id][uq["qid"]]
            chosen_idx = choices_dict.get(uq["model_choice"], 0) # default index 0 when gpt3 had a failure.
            gold_idx = choices_dict.get(uq["user_choice"], 0)
            is_collapsed_correct = collapsed_class(choices=choices, choice=choices[chosen_idx]) == \
                                   collapsed_class(choices=choices, choice=choices[gold_idx])
            evals.append(int(is_correct))
            evals_collapsed.append(int(is_collapsed_correct))

    overall_accuracy = sum(evals)/ len(evals)
    wilson_overall_accuracy = wilson(p=overall_accuracy, n=len(evals))

    collapsed_overall_accuracy = sum(evals_collapsed)/ len(evals_collapsed)
    wilson_collapsed_accuracy = wilson(p=collapsed_overall_accuracy, n=len(evals_collapsed))

    p = f"\nAverage accuracy from {predictions_evaluated_json_fp}:\n" \
        f"Exact match: {overall_accuracy: 0.3f} +/- {overall_accuracy - wilson_overall_accuracy[0] :0.3f} -> {wilson_overall_accuracy}\n" \
        f"Collapsed  : {collapsed_overall_accuracy: 0.3f} +/- {collapsed_overall_accuracy - wilson_collapsed_accuracy[0] :0.3f}  -> {wilson_collapsed_accuracy}"
    print(p)



def main_topk():
    mypath = "data/model-output-topk"
    user_responses_json_fp="data/opinionqa/sampled_user_responses_20_decl_topk_compressed.jsonl"

    for file in [f for f in os.listdir(mypath) if isdir(join(mypath, f))]:
        eval_fp = join(mypath, file, "model_generation.json")
        # eval_fp="data/model-output-topk/imp-3pts_exp-t-1-u35-q30/model_generation.json")
        if os.path.exists(eval_fp):
            compute_collapsed_accuracy(user_responses_json_fp=user_responses_json_fp, predictions_evaluated_json_fp=eval_fp)


if __name__ == '__main__':
    main()
    main_topk()