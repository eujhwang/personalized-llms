import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd
from gptinference.utils import read_jsonl_or_json, write_json


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="data/opinionqa/sampled_user_responses_decl.json", help="synthesized opinionqa json path")
    parser.add_argument("--out_path", type=str, default="data/model-output/topicwise_accuracy.json", help="output metrics json path")
    parser.add_argument("--model_outputs_path", type=str, default="data/model-output", help="model output directory json path")
    args = parser.parse_args()

    # add topic-wise accuracy when there are multiple dirs
    model_output_files = {
                          "no-persona-t-1-u35-q30": "no-persona",
                          "explicit-t-1-u35-q30": "explicit",
                          # "imp-8pts_exp-t-1-u35-q30": "explicit+imp8",
                          "imp-16pts_exp-t-1-u35-q30": "explicit+imp16",
                          # "implicit_8pts-t-1-u35-q30": "implicit8",
                          # "implicit_16pts-t-1-u35-q30": "implicit16",
                         }
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

