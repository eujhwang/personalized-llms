import argparse
import os
from pathlib import Path
from typing import List

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
    parser.add_argument("--in_path", type=str, default="data/opinionqa/sampled_user_responses_decl.json", help="json path")
    parser.add_argument("--out_dir", type=str, default="data/model-output/", help="json path")
    args = parser.parse_args()

    # add topic-wise accuracy when there are multiple dirs
    metrics_files = [os.path.join(args.out_dir, "implicit_2pts-t2-u2-q2/model_accuracy.json"),
                     os.path.join(args.out_dir, "implicit_1pts-t2-u2-q1/model_accuracy.json")]

    topic_metrics = calculate_topicwise_accuracy(args.in_path, metrics_files)
    topic_metrics_file = os.path.join(args.out_dir, "model_topic_accuracy.json")
    write_json(outpath=topic_metrics_file, json_data=topic_metrics)