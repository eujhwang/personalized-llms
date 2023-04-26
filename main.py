import itertools
import json
import math
import random

import torch.cuda
import argparse
import numpy as np
import pandas as pd
import os
import ast
from sklearn.model_selection import train_test_split
from src.aggregate_pairs import create_demo_to_qa_dict, find_pair_by_demographic, calculate_cohen_kappa_score, \
    calculate_cohen_kappa_score_per_subtopic, find_pair_by_topic, aggregate_responses_by_topic, split_data
from src.utils import set_seed

PEW_SURVEY_LIST = [26, 27, 29, 32, 34, 36, 41, 42, 43, 45, 49, 50, 54, 82, 92]

pd.options.display.max_columns = 15
# pd.options.display.max_rows = 999

explicit_info = [
    "CREGION", "AGE", "SEX", "EDUCATION", "CITIZEN", "MARITAL", "RELIG", "RELIGATTEND", "POLPARTY", "INCOME",
    "POLIDEOLOGY", "RACE"
]


def load_question_info(info_df):
    info_keys = info_df['key'].tolist()
    info_questions = info_df['question'].tolist()
    info_choices = info_df['references'].tolist()
    info_dict = {}
    for key, question, choice in zip(info_keys, info_questions, info_choices):
        info_dict[key] = {
            "question": question,
            "choice": choice
        }
    return info_dict

def process_implicit_responses(info_keys, resp_df):
    resp_implicit_dict = {}
    total_implicit_len = 0
    for info_key in info_keys:
        data_list = resp_df[info_key].tolist()
        resp_implicit_dict[info_key] = data_list  # 'SAFECRIME_W26': ['Very safe', 'Not too safe', 'Very safe',...]
        total_implicit_len = len(data_list)

    return resp_implicit_dict, total_implicit_len


def process_explicit_responses(meta_keys, resp_df):
    resp_explicit_dict = {}
    total_explicit_len = 0
    for meta_key in meta_keys:
        data_list = resp_df[meta_key].tolist()
        resp_explicit_dict[meta_key] = data_list
        total_explicit_len = len(data_list)
    # print("resp_explicit_dict:", resp_explicit_dict.keys())

    return resp_explicit_dict, total_explicit_len


def extract_sub_topic(user_resp_list):
    # load topic-mapping.npy
    topic_mapping = np.load('data/topic_mapping.npy', allow_pickle=True)
    topic_mapping = topic_mapping.tolist()
    print(len(topic_mapping), type(topic_mapping), )

    for i, resp in enumerate(user_resp_list[:5]):
        print(f"###################### {i} ######################")
        implicit_info_dict = resp['implicit_info']
        for question, answer in implicit_info_dict.items():
            sub_topic = topic_mapping[question]
            print("sub_topic:", sub_topic)

    # for i, (key, value) in enumerate(topic_mapping.items()):
    #     print("key:", key)
    #     print("value:", value)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load topic-mapping.npy
    topic_mapping = np.load('data/topic_mapping.npy', allow_pickle=True)
    topic_mapping = topic_mapping.tolist()
    print(len(topic_mapping), type(topic_mapping), )


    DATASET_DIR = os.path.join(args.data_dir, 'human_resp')
    RESULT_DIR = os.path.join(args.data_dir, 'runs')

    SURVEY_LIST = [f'American_Trends_Panel_W{SURVEY_WAVE}' for SURVEY_WAVE in PEW_SURVEY_LIST]
                  # + ['Pew_American_Trends_Panel_disagreement_500']

    SURVEY_LIST = SURVEY_LIST #[:1]
    print("SURVEY_LIST:", len(SURVEY_LIST), SURVEY_LIST)

    all_user_responses_file = "data/opinionqa/all_user_responses.json"
    all_qinfo_file = "data/opinionqa/all_qinfo_dict.json"
    all_demographic_file = "data/opinionqa/all_demographic_dict.json"
    if not (os.path.exists(all_user_responses_file) and os.path.exists(all_qinfo_file) and os.path.exists(all_demographic_file)):
        total_responses = 0
        all_qinfo_dict = {}
        all_demographic_dict = {}
        # train_data_dict, val_data_dict, test_data_dict = {}, {}, {}
        all_user_responses = []
        user_id = 0
        for SURVEY_NAME in SURVEY_LIST:
            print("############################", SURVEY_NAME, "############################")
            qinfo_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'info.csv'))
            meta_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'metadata.csv'))
            resp_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'responses.csv'), engine='python')

            #### info_df processing ####
            qinfo_dict = load_question_info(qinfo_df)
            qinfo_keys = qinfo_dict.keys()
            # print("qinfo_dict:", len(qinfo_dict), qinfo_dict)
            all_qinfo_dict.update(qinfo_dict)

            #### metadata df processing ####
            meta_keys = meta_df['key'].tolist()
            # print("meta_keys:", meta_keys)

            #### resp_df processing ##
            user_ids = resp_df['QKEY'].tolist()

            resp_implicit_dict, total_implicit_len = process_implicit_responses(qinfo_keys, resp_df)
            resp_explicit_dict, total_explicit_len = process_explicit_responses(meta_keys, resp_df)
            total_len = len(resp_df)
            total_responses += total_len
            assert total_implicit_len == total_explicit_len == total_len
            print("total_implicit_len", total_implicit_len, "total_explicit_len", total_explicit_len, len(user_ids))

            for i in range(total_len):
                user_resp_dict = {}
                implicit_dict = {}
                for q_key in qinfo_keys:
                    response = resp_implicit_dict[q_key][i]  # list of responses
                    if isinstance(response, float) and math.isnan(response):
                        continue

                    question = qinfo_dict[q_key]['question']
                    choices = qinfo_dict[q_key]['choice']
                    choices = ast.literal_eval(choices)
                    choices = "/".join(choices)

                    topic_mapping_key = f"{question} [{choices}]"
                    implicit_info = {
                        "question": qinfo_dict[q_key]['question'],
                        "choice": qinfo_dict[q_key]['choice'],
                        "answer": response,
                        "question_id": q_key,
                        "subtopic_fg": topic_mapping[topic_mapping_key]['fg'],
                        "subtopic_cg": topic_mapping[topic_mapping_key]['cg'],
                    }
                    implicit_dict[q_key] = implicit_info

                explicit_dict = {}
                for key in resp_explicit_dict.keys():
                    explicit_dict.update({
                        key: resp_explicit_dict[key][i]
                    })
                all_demographic_dict[user_id] = explicit_dict
                user_resp_dict['user_id'] = user_id
                user_resp_dict['survey_name'] = SURVEY_NAME
                user_resp_dict['implicit_info'] = implicit_dict
                user_resp_dict['explicit_info'] = explicit_dict
                user_id += 1

                all_user_responses.append(user_resp_dict)

        with open(all_user_responses_file, "w") as f:
            json.dump(all_user_responses, f, indent=4)
        with open(all_qinfo_file, "w") as f:
            json.dump(all_qinfo_dict, f, indent=4)
        with open(all_demographic_file, "w") as f:
            json.dump(all_demographic_dict, f, indent=4)

    else:
        with open(all_user_responses_file, "r") as fd:
            print("loading all_user_responses...")
            all_user_responses = json.load(fd)
            print(len(all_user_responses))

        with open(all_qinfo_file, "r") as fd:
            all_qinfo_dict = json.load(fd)
            print("all_qinfo_dict", len(all_qinfo_dict))

        with open(all_demographic_file, "r") as fd:
            all_demographic_dict = json.load(fd)
            print("all_demographic_dict", len(all_demographic_dict))

    demographic_pair_dict_path = "demographic_pair_dict.json"
    if not os.path.exists(demographic_pair_dict_path):
        demographic_pair_dict = find_pair_by_demographic(all_user_responses)
    else:
        with open(demographic_pair_dict_path, "r") as fd:
            print("loading demographic_pair_dict...")
            demographic_pair_dict = json.load(fd)
            print(len(demographic_pair_dict))

    calculate_cohen_kappa_score(demographic_pair_dict)

    topic_pair_dict_path = "topic_pair_dict.json"
    if not os.path.exists(topic_pair_dict_path):
        topic_pair_dict = find_pair_by_topic(all_user_responses)
    else:
        with open(topic_pair_dict_path) as fd:
            print("loading topic_pair_dict...")
            topic_pair_dict = json.load(fd)
            print(len(topic_pair_dict))

    calculate_cohen_kappa_score_per_subtopic(topic_pair_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data/opinion-qa/', help="start collecting memes from reddit")
    parser.add_argument("--create_demo_dict", action='store_true', help="create demographic to qa pair dict")
    parser.add_argument("--create_split", action='store_true', help="create split of val and test")

    args = parser.parse_args()
    set_seed(42)
    main(args)
