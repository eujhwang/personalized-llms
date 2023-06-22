import argparse
import ast
import json
import os
import random

import numpy as np
import pandas as pd
from gptinference.utils import read_jsonl_or_json
from scipy.stats import wasserstein_distance

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


def set_seed(seed):
    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    os.environ['PYTHONHASHSEED'] = str(seed)

def extract_human_opinions(user_responses_jsons):

    for user_resp in user_responses_jsons:
        # ['user_id', 'survey', 'topic', 'explicit_persona', 'implicit_persona', 'implicit_questions']
        user_id = user_resp["user_id"]
        survey = user_resp["survey"]
        topic = user_resp["topic"]
        explicit_persona = user_resp["explicit_persona"]
        implicit_persona = user_resp["implicit_persona"]
        implicit_questions = user_resp["implicit_questions"]

        print("user_id: {}, survey: {}, topic: {}, explicit_persona: {}, implicit_persona: {}".format(
            user_id, survey, topic, explicit_persona, implicit_persona
        ))
        # print("implicit_questions: {}".format(implicit_questions))
        for imp_que in implicit_questions:
            # print("imp_que:", imp_que.keys())
            qid = imp_que["qid"]
            question = imp_que["question"]
            choices = imp_que["choices"]
            answer = imp_que["answer"]
            subtopic_cg = imp_que["subtopic_cg"]
        break


def extract_model_opinions(model_generations, survey_name, qinfo_dict, question_keys):
    """
create model dataframe with log probability

columns:
user_id survey_name qid answer  answer_choice   log_prob    prob

    """



    # model_df = pd.DataFrame(columns=["user_id", "survey_name", "qid", "question", "choices",
    #                                  "model_answer", "model_choice", "log_prob", "prob"])
    model_df_list = []
    for model_gen in model_generations:
        # print("model_gen: {}".format(model_gen))
        if SURVEY_TO_TOPIC[survey_name] != model_gen["topic"]:
            continue

        user_id = model_gen["user_id"]
        topic = model_gen["topic"]
        generations = model_gen["generated_output"]
        for gen in generations:
            qid = gen["qid"]
            model_choice = gen["model_choice"]
            if model_choice == "UNKNOWN" or len(model_choice) > 1:
                continue
            log_prob = gen["response"]["choices"][0]["logprobs"]["token_logprobs"][0]
            prob = np.exp(log_prob)
            question = qinfo_dict[qid]["question"]
            choices = ast.literal_eval(qinfo_dict[qid]["choice"])

            # print("qid: {}, question: {}, choices: {}".format(qid, question, choices))
            # print("model_choice: {}, log_prob: {}, prob: {}".format(model_choice, log_prob, prob))
            # qid_list.append(qid)
            #
            gen_dict = {
                "user_id": user_id,
                "survey_name": survey_name,
                "qid": qid,
                "question": question,
                "choices": choices,
                "model_answer": choices[OUTPUT_MAP.index(model_choice)],
                "model_choice": model_choice,
                "log_prob": log_prob,
                "prob": prob,
            }

            # model_df = pd.DataFrame([model_df])
            model_df_list.append(pd.DataFrame([gen_dict]))
        # print("model_df:", len(model_df), model_df.head())
        # assert False
    model_df = pd.concat(model_df_list, ignore_index=True, axis=0)

    # print("model_df:", len(model_df), model_df.head())
    # assert False
    return model_df


def get_max_wd(model_ordinal, human_ordinal):
    d0, d1 = np.zeros(len(model_ordinal)), np.zeros(len(human_ordinal))
    d0[np.argmax(model_ordinal)] = 1
    d1[np.argmin(human_ordinal)] = 1
    max_wd = wasserstein_distance(model_ordinal, human_ordinal, d0, d1)
    return max_wd

# def get_max_wd(ordered_ref_weights):
#     d0, d1 = np.zeros(len(ordered_ref_weights)), np.zeros(len(ordered_ref_weights))
#     d0[np.argmax(ordered_ref_weights)] = 1
#     d1[np.argmin(ordered_ref_weights)] = 1
#     max_wd = wasserstein_distance(ordered_ref_weights, ordered_ref_weights, d0, d1)
#     return max_wd

def main(args):
    print()
    data_dir = "../data/opinion-qa/human_resp/"
    file_path = "../data/opinionqa/sampled_user_responses_20.json"
    user_responses_jsons = read_jsonl_or_json(file_path)
    # resp_file = "../data/opinion-qa/human_resp/American_Trends_Panel_W26/responses.csv"
    # info_file = "../data/opinion-qa/human_resp/American_Trends_Panel_W26/info.csv"
    # # RESULT_DIR = './data/runs'
    # # RESULT_FILES = [f for f in os.listdir(RESULT_DIR) if SURVEY_NAME in f and f'context={CONTEXT}' in f]
    # run_dir = "../data/opinion-qa/runs/"
    # model_dir = "opinions_qa:survey=Pew_American_Trends_Panel_W26,num_logprobs=100,context=default,num_train_trials=1,model=openai_text-davinci-003,num_train_trials=1" #"scenario_state.json"
    # imp-16pts_exp-t-1-u35-q30, no-persona-t-1-u35-q30, explicit-t-1-u35-q30
    model_out = "../data/model-output-prob/text-davinci-003/imp-16pts_exp-t-1-u35-q30/model_generation.json"
    # ["imp-3pts_exp-t-1-u35-q30", "imp-8pts_exp-t-1-u35-q30", "imp-8pts_exp-t-1-u35-q30-explicit-means-demo",
    #  "imp-8pts_exp-t-1-u35-q30-explicit-means-ideo", "implicit_3pts-t-1-u35-q30", "implicit_8pts-t-1-u35-q30"]
    model_out = "../data/model-output-prob-topk/text-davinci-003/implicit_8pts-t-1-u35-q30/model_generation.json"
    model_generations = read_jsonl_or_json(model_out)
    qinfo_dict = read_jsonl_or_json("../data/opinionqa/all_qinfo_dict.json")


    # selected_rows = []
    # question_keys = []
    # for user_resp in user_responses_jsons:
    #     user_id = user_resp["user_id"]
    #     selected_rows.append(user_id)
    #     implicit_questions = user_resp["implicit_questions"]
    #     for imp_que in implicit_questions:
    #         question_keys.append(imp_que["qid"])
    #         # assert False
    # question_keys = list(set(question_keys))

    PEW_SURVEY_LIST = [26, 27, 29, 32, 34, 36, 41, 42, 43, 45, 49, 50, 54, 82, 92]
    SURVEY_LIST = [f'American_Trends_Panel_W{SURVEY_WAVE}' for SURVEY_WAVE in PEW_SURVEY_LIST]


    init_num = 0
    count = 0
    overall_alignment_score = []
    for SURVEY_WAVE in PEW_SURVEY_LIST:
        SURVEY_NAME = f'American_Trends_Panel_W{SURVEY_WAVE}'

        selected_rows = []
        question_keys = []
        for user_resp in user_responses_jsons:
            survey = user_resp["survey"]
            if survey != SURVEY_NAME:
                continue
            user_id = user_resp["user_id"] - init_num
            implicit_questions = user_resp["implicit_questions"]
            selected_rows.append(user_id)
            for imp_que in implicit_questions:
                question_keys.append(imp_que["qid"])
        question_keys = set(question_keys)

        model_df = extract_model_opinions(model_generations, SURVEY_NAME, qinfo_dict, question_keys)
        model_question_keys = set(model_df['qid'].tolist())

        inter_question_keys = list(question_keys & model_question_keys)
        # print("question_keys: {}, model_question_keys: {}, inter_question_keys: {}".format(len(question_keys), len(model_question_keys), len(inter_question_keys)))

        assert len(model_question_keys) == len(inter_question_keys)

        survey_path = os.path.join(data_dir, SURVEY_NAME, "responses.csv")
        info_path = os.path.join(data_dir, SURVEY_NAME, "info.csv")

        survey_df = pd.read_csv(survey_path)
        selected_df = survey_df.loc[survey_df.index[selected_rows]]

        info_df = pd.read_csv(info_path)
        key_to_ordering = {k: v for k, v in zip(info_df['key'], info_df['option_ordinal'])}
        # print("selected_df: {}", selected_df.head())

        weight_key = [w for w in selected_df.columns if w == f'WEIGHT_W{SURVEY_WAVE}']
        assert len(weight_key) == 1
        weight_key = weight_key[0]
        # weight_key = f'WEIGHT_W{SURVEY_WAVE}'

        # print("weight_key:", len(weight_key), weight_key)
        # for each qkey, aggregate weights per answer choice
        alignment_score_list = []
        for qkey in sorted(inter_question_keys):
            choices = ast.literal_eval(qinfo_dict[qkey]["choice"])

            model_weight_df = model_df[model_df['qid'] == qkey]
            model_weight_df = model_weight_df[[type(v) == str for v in model_weight_df['model_choice']]]
            model_weight_df = model_weight_df.groupby(['model_answer'], as_index=False).agg({'prob': sum})
            # print("model_weight_df:", model_weight_df)

            weight_df = selected_df[[weight_key, qkey]]
            weight_df = weight_df[[type(v) == str for v in weight_df[qkey]]] # exclude Nan answers
            weight_df = weight_df.groupby([qkey], as_index=False).agg({weight_key: sum}) # aggregate weights per answer choice
            # print("weight_df3: {}".format(weight_df.head()))

            ordinal_mapping = ast.literal_eval(key_to_ordering[qkey])
            # print("choices:", choices, "ordinal_mapping:", ordinal_mapping)
            # {'Overall': {'Essential': 17.4595796, 'Important but not essential': 11.0031167, 'Not important': 3.5154201, 'Should not be done': 3.5080921}}
            model_dist_all = {k: v for k, v in zip(model_weight_df['model_answer'], model_weight_df['prob'])}
            dist_all = {k: v for k, v in zip(weight_df[qkey], weight_df[weight_key])}
            human_opinion_distribution, model_opinion_distribution = [], []
            human_ordinal, model_ordinal = [], []
            for i, choice in enumerate(choices):
                if choice.lower() == "refused":
                    continue
                if i >= len(ordinal_mapping): # refusal
                    break
                    # human_ordinal.append(ordinal_mapping[i])
                    # model_ordinal.append(ordinal_mapping[i])
                else:
                    if choice in dist_all:
                        human_opinion_distribution.append(dist_all[choice])
                    else:
                        human_opinion_distribution.append(0)
                    human_ordinal.append(ordinal_mapping[i])

                    if choice in model_dist_all:
                        model_opinion_distribution.append(model_dist_all[choice])
                    else:
                        model_opinion_distribution.append(0)
                    model_ordinal.append(ordinal_mapping[i])

            human_opinion_distribution = np.array(human_opinion_distribution)
            human_opinion_distribution /= np.sum(human_opinion_distribution)
            # print("human_opinion_distribution:", human_opinion_distribution)

            if np.sum(model_opinion_distribution) == 0:
                count += 1
                continue
            model_opinion_distribution = np.array(model_opinion_distribution)
            model_opinion_distribution /= np.sum(model_opinion_distribution)
            # print("model_opinion_distribution:", model_opinion_distribution)

            human_ordinal = np.array(human_ordinal)
            model_ordinal = np.array(model_ordinal)
            # print("human_ordinal:", human_ordinal, "model_ordinal:", model_ordinal)
            wd = wasserstein_distance(model_ordinal, human_ordinal, model_opinion_distribution, human_opinion_distribution) # minimum distance
            max_wd = get_max_wd(model_ordinal, human_ordinal)
            if max_wd == 0:
                continue
            alignment_score = 1 - (wd / max_wd)
            alignment_score_list.append(alignment_score)
            assert (wd/max_wd) <= 1

            # assert False
        print("alignment_score:", sum(alignment_score_list)/len(alignment_score_list))
        alignment_score = sum(alignment_score_list)/len(alignment_score_list)
        overall_alignment_score.append(alignment_score)
        # assert False
        init_num += len(survey_df)
    overall_alignment_score = sum(overall_alignment_score)/len(overall_alignment_score)
    print("count: {}".format(count))
    print("overall_alignment_score:", overall_alignment_score)
    # no_persona: 0.6585276772862255 (0.66),
    # explicit 0.7577939159119425 (0.76),
    # imp+exp: 0.7742751225971694 (0.77)
    # ["imp-3pts_exp-t-1-u35-q30": 0.7925126308181667 (0.79),
    # "imp-8pts_exp-t-1-u35-q30": 0.7906115637940622 (0.79),
    # "imp-8pts_exp-t-1-u35-q30-explicit-means-demo": 0.7840321628910257 (0.78),
    #  "imp-8pts_exp-t-1-u35-q30-explicit-means-ideo": 0.7890318447388676 (0.79),
    #  "implicit_3pts-t-1-u35-q30": 0.7737607822189048 (0.77),
    #  "implicit_8pts-t-1-u35-q30": 0.7744019071170298 (0.77)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='', choices=["eg", "anlg"], help="dataset: eg or anlg")
    args = parser.parse_args()
    set_seed(42)
    main(args)