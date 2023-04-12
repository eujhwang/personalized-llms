import itertools
import json
import math

import torch.cuda
import argparse
import numpy as np
import pandas as pd
import os

from src.utils import extract_human_opinions, DEMOGRAPHIC_ATTRIBUTES

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
    # for i, (key, value) in enumerate(topic_mapping.items()):
    #     print("key:", key)
    #     print("value:", value)
    #
    #     if i >= 3:
    #         break

    DATASET_DIR = os.path.join(args.data_dir, 'human_resp')
    RESULT_DIR = os.path.join(args.data_dir, 'runs')

    SURVEY_LIST = [f'American_Trends_Panel_W{SURVEY_WAVE}' for SURVEY_WAVE in PEW_SURVEY_LIST]
                  # + ['Pew_American_Trends_Panel_disagreement_500']

    SURVEY_LIST = SURVEY_LIST #[:1]
    print("SURVEY_LIST:", len(SURVEY_LIST), SURVEY_LIST)

    resp_indi_dict = {}
    total_responses = 0
    for SURVEY_NAME in SURVEY_LIST:
        print("############################", SURVEY_NAME, "############################")
        qinfo_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'info.csv'))
        meta_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'metadata.csv'))
        resp_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'responses.csv'), engine='python')

        #### info_df processing ####
        qinfo_dict = load_question_info(qinfo_df)
        qinfo_keys = qinfo_dict.keys()
        print("qinfo_dict:", len(qinfo_dict), qinfo_dict)

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
                # print("check!", len(resp_indi_dict[key]["implicit_info"]))

    demo_to_question_dict = {}
    for i, key in enumerate(resp_indi_dict.keys()):
        implicit_info = resp_indi_dict[key]["implicit_info"]
        if len(implicit_info) <= 1:
            continue

        # get all question keys that exist in different responses
        question_keys = set()
        for info in implicit_info:
            question_keys.update(info.keys())

        question_dict = {}
        for q_key in list(question_keys):
            if q_key not in question_dict.keys():
                question_dict[q_key] = []

            for info in implicit_info:
                if q_key not in info.keys():
                    continue
                answer = info[q_key]["answer"]
                question_dict[q_key].append(answer)
        demo_to_question_dict[key] = question_dict

        if i >=300:
            break

    # print("demo_to_question_dict:", demo_to_question_dict)
    new_demo_to_question_dict = {}
    demo_qa_num = []
    for key, qa_pair in demo_to_question_dict.items():
        demo_key = key
        demo_q_list = []
        for q, a in qa_pair.items():
            if len(a) <= 1:
                continue
            demo_q_list.append({q: a})
        if len(demo_q_list) > 0:
            new_demo_to_question_dict[demo_key] = demo_q_list
            demo_qa_num.append(len(demo_q_list))
        # print()
    # print("avg_demo_qa_num:", sum(demo_qa_num)/len(demo_qa_num))


    for i, (demo_key, demo_q_list) in enumerate(new_demo_to_question_dict.items()):
        print()
        agreement = 0
        disagreement = 0
        for demo_q in demo_q_list:
            for k, v in demo_q.items():
                answers = v
                for combi in list(itertools.combinations(answers, 2)):
                    if combi[0] == combi[1]:
                        agreement += 1
                    else:
                        disagreement += 1
        print("demo_q_list:", len(demo_q_list), "agreement:", agreement, "disagreement:", disagreement)


    # print("new_demo_to_question_dict:", len(new_demo_to_question_dict), new_demo_to_question_dict)



    # print("total_keys:", len(resp_indi_dict.keys()))
    # print("total_responses:", total_responses)
    # print("total_pairs:", len(pairs))


    # with open("resp_indi_dict.json", "w") as f:
    #     json.dump(resp_indi_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data/opinion-qa/', help="start collecting memes from reddit")
    args = parser.parse_args()

    main(args)
