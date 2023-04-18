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

DEMO_MAP = {
    "CREGION": "Region",
    "AGE": "Age",
    "SEX": "Gender",
    "EDUCATION": "Education",
    "CITIZEN": "Citizenship",
    "MARITAL": "Marital status",
    "RELIG": "Religion",
    "RELIGATTEND": "Frequency of religious attendance",
    "POLPARTY": "Political party",
    "INCOME": "Income",
    "POLIDEOLOGY": "Political ideology",
    "RACE": "Race",
}


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

    all_user_responses_file = "all_user_responses.json"
    all_qinfo_file = "all_qinfo_dict.json"
    all_demographic_file = "all_demographic_dict.json"
    # if not (os.path.exists(all_user_responses_file) and os.path.exists(all_qinfo_file) and os.path.exists(all_demographic_file)):
    total_responses = 0
    all_qinfo_dict = {}
    all_demographic_dict = {}
    # train_data_dict, val_data_dict, test_data_dict = {}, {}, {}
    all_user_responses = []
    sampled_responses = []
    prev_user_id = 0
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
        # user_ids = resp_df['QKEY'].tolist()

        resp_implicit_dict, total_implicit_len = process_implicit_responses(qinfo_keys, resp_df)
        resp_explicit_dict, total_explicit_len = process_explicit_responses(meta_keys, resp_df)
        total_users = len(resp_df)
        total_responses += total_users
        assert total_implicit_len == total_explicit_len == total_users
        print("total_implicit_len", total_implicit_len, "total_explicit_len", total_explicit_len)

        sampled_user_ids = sorted(random.sample(range(total_users), 105))
        # print("sampled_user_ids:", len(sampled_user_ids), sampled_user_ids)

        for user_id in sampled_user_ids[:100]:
            implicit_data = []
            for q_key in sorted(qinfo_keys):
                response = resp_implicit_dict[q_key][user_id]  # list of responses
                if isinstance(response, float) and math.isnan(response):
                    continue

                question = qinfo_dict[q_key]['question']
                choices = qinfo_dict[q_key]['choice']
                choices = ast.literal_eval(choices)
                choices = [choice for choice in choices]
                topic_mapping_key = f"{question} [{'/'.join(choices)}]"
                # {'question': 'How safe, if at all, would you say your local community is from crime? Would you say it is',
                # 'choices': ['Very safe', 'Somewhat safe', 'Not too safe', 'Not at all safe', 'Refused'],
                # 'answer': 'Not too safe'}
                res = {
                    "qid": q_key,
                    "question": question,
                    "choices": choices,
                    "answer": response,
                    "subtopic_cg": topic_mapping[topic_mapping_key]['cg'],
                }
                implicit_data.append(res)

            implicit_persona, implicit_questions = train_test_split(implicit_data, random_state=42, test_size=0.8)

            explicit_data = []
            for demo in sorted(resp_explicit_dict.keys()):
                demo_key = DEMO_MAP[demo]
                demo_value = resp_explicit_dict[demo][user_id]
                # print("demo_key:", demo, DEMO_MAP[demo], "info:", demo_value)
                if isinstance(demo_value, float) and math.isnan(demo_value):
                    continue
                explicit_data.append({demo_key: demo_value})

            sampled_responses.append({
                "user_id": prev_user_id + user_id,
                "survey": SURVEY_NAME,
                "topic": SURVEY_TO_TOPIC[SURVEY_NAME],
                "explicit_persona": explicit_data,
                "implicit_persona": implicit_persona,
                "implicit_questions": implicit_questions,
            })

        prev_user_id += total_users

    with open("sampled_user_responses.json", "w") as f:
        json.dump(sampled_responses, f, indent=4)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data/opinion-qa/', help="start collecting memes from reddit")
    parser.add_argument("--create_demo_dict", action='store_true', help="create demographic to qa pair dict")
    parser.add_argument("--create_split", action='store_true', help="create split of val and test")

    args = parser.parse_args()
    set_seed(42)
    main(args)
