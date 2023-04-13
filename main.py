import itertools
import json
import math

import torch.cuda
import argparse
import numpy as np
import pandas as pd
import os
import ast
from sklearn.model_selection import train_test_split
from src.aggregate_pairs import create_demo_to_qa_dict
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


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load topic-mapping.npy
    topic_mapping = np.load('data/topic_mapping.npy', allow_pickle=True)
    topic_mapping = topic_mapping.tolist()
    # print(len(topic_mapping), type(topic_mapping), )
    #
    for i, (key, value) in enumerate(topic_mapping.items()):
        print("key:", key)
        print("value:", value)

        if i >= 3:
            break

    DATASET_DIR = os.path.join(args.data_dir, 'human_resp')
    RESULT_DIR = os.path.join(args.data_dir, 'runs')

    SURVEY_LIST = [f'American_Trends_Panel_W{SURVEY_WAVE}' for SURVEY_WAVE in PEW_SURVEY_LIST]
                  # + ['Pew_American_Trends_Panel_disagreement_500']

    SURVEY_LIST = SURVEY_LIST #[:1]
    print("SURVEY_LIST:", len(SURVEY_LIST), SURVEY_LIST)

    resp_indi_dict = {}
    total_responses = 0
    all_qinfo_dict = {}
    train_data_dict, val_data_dict, test_data_dict = {}, {}, {}
    for SURVEY_NAME in SURVEY_LIST:
        print("############################", SURVEY_NAME, "############################")
        qinfo_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'info.csv'))
        meta_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'metadata.csv'))
        resp_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'responses.csv'), engine='python')

        #### info_df processing ####
        qinfo_dict = load_question_info(qinfo_df)
        qinfo_keys = qinfo_dict.keys()
        print("qinfo_dict:", len(qinfo_dict), qinfo_dict)
        all_qinfo_dict.update(qinfo_dict)

        #### metadata df processing ####
        meta_keys = meta_df['key'].tolist()
        print("meta_keys:", meta_keys)

        #### resp_df processing ##
        user_ids = resp_df['QKEY'].tolist()

        resp_implicit_dict, total_implicit_len = process_implicit_responses(qinfo_keys, resp_df)
        resp_explicit_dict, total_explicit_len = process_explicit_responses(meta_keys, resp_df)
        total_len = len(resp_df)
        total_responses += total_len
        assert total_implicit_len == total_explicit_len == total_len
        print("total_implicit_len", total_implicit_len, "total_explicit_len", total_explicit_len, len(user_ids))

        if args.create_demo_dict:
            """
            {
            "demographic information": { 
                "implicit_info": {
                        "question": question,
                        "choice": choices,
                        "answer": user response,
                        "question_id": question id
                    }
            }
            """
            for i in range(total_len):
                explicit_info_dict = {}
                for meta_key in meta_keys:
                    explicit_info_dict[meta_key] = resp_explicit_dict[meta_key][i]  # list of responses

                key = tuple(sorted(explicit_info_dict.items()))

                implicit_info_dict = {}
                for info_key in qinfo_keys:
                    response = resp_implicit_dict[info_key][i]  # list of responses
                    if isinstance(response, float) and math.isnan(response):
                        continue
                    implicit_info = {
                        "question": qinfo_dict[info_key]['question'],
                        "choice": qinfo_dict[info_key]['choice'],
                        "answer": response,
                        "question_id": info_key
                    }
                    implicit_info_dict[info_key] = implicit_info

                if key not in resp_indi_dict.keys():
                    resp_indi_dict[key] = {
                        "implicit_info": [implicit_info_dict]
                    }
                else:
                    resp_indi_dict[key]["implicit_info"].append(implicit_info_dict)

        # Make a split of the users into two group: dev and test. Each user datapoint should contain all the info
        if args.create_split:

            print("resp_implicit_dict:", resp_implicit_dict.keys())
            print("resp_explicit_dict:", resp_explicit_dict.keys())
            user_resp_list = []
            for i in range(total_len):
                user_resp_dict = {}
                for key in resp_implicit_dict.keys():
                    response = resp_implicit_dict[key][i]
                    if isinstance(response, float) and math.isnan(response):
                        continue

                    question = qinfo_dict[key]['question']
                    choices = qinfo_dict[key]['choice']
                    choices = ast.literal_eval(choices)
                    choices = "/".join(choices)
                    user_resp_dict.update({
                        f"{question} [{choices}]": response
                    })

                for key in resp_explicit_dict.keys():
                    user_resp_dict.update({
                        key: resp_explicit_dict[key][i]
                    })
                user_resp_list.append(user_resp_dict)

            x_train ,x_test = train_test_split(user_resp_list, random_state=42, test_size=0.3)
            x_val, x_test = train_test_split(x_test, random_state=42, test_size=0.5)
            
            train_data_dict[SURVEY_NAME] = x_train
            val_data_dict[SURVEY_NAME] = x_val
            test_data_dict[SURVEY_NAME] = x_test


    if args.create_demo_dict:
        #  statistics where we show two users having exactly same demographics have different opinion answers for a same question
        demo_to_question_dict = create_demo_to_qa_dict(resp_indi_dict)
        with open("demo_to_question_dict.json", "w") as f:
            json.dump(demo_to_question_dict, f, indent=4)

    if args.create_split:
        with open("train_data.json", "w") as f:
            json.dump(train_data_dict, f, indent=4)

        with open("val_data.json", "w") as f:
            json.dump(val_data_dict, f, indent=4)

        with open("test_data.json", "w") as f:
            json.dump(test_data_dict, f, indent=4)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data/opinion-qa/', help="start collecting memes from reddit")
    parser.add_argument("--create_demo_dict", action='store_true', help="create demographic to qa pair dict")
    parser.add_argument("--create_split", action='store_true', help="create split of val and test")

    args = parser.parse_args()
    set_seed(42)
    main(args)
