import ast
import itertools
import json
import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

survey_name_map = {
                      'American_Trends_Panel_W26': "Guns",
                      'American_Trends_Panel_W27': "Automation vehicles",
                  'American_Trends_Panel_W29': "Views on gender",
                  'American_Trends_Panel_W32': "Community types&sexual harassment",
                  'American_Trends_Panel_W34': "Biomedical&food issues",
                  'American_Trends_Panel_W36': "Gender&Leadership",
                  'American_Trends_Panel_W41': "America in 2050",
                  'American_Trends_Panel_W42': "Trust in science",
                  'American_Trends_Panel_W43': "Race",
                  'American_Trends_Panel_W45': "Misinformation95",
                  'American_Trends_Panel_W49': "Privacy&Surveilance",
                  'American_Trends_Panel_W50': "Family&Relationships",
                  'American_Trends_Panel_W54': "Economic inequality",
                  'American_Trends_Panel_W82': "Global attitudes",
                  'American_Trends_Panel_W92': "Political views"
}

def get_topic_mapping_key(question, choices):
    choices = ast.literal_eval(choices)
    choices = "/".join(choices)

    topic_mapping_key = f"{question} [{choices}]"
    return topic_mapping_key


def visualize_demo_to_agreement_score(demo_key_list, demo_score_list):
    # Generate 1114 random points
    # np.random.seed(0)
    x = np.arange(len(demo_key_list))
    y = np.array(demo_score_list)

    # Create a scatter plot
    plt.scatter(x, y, c='blue', marker='o', edgecolors='black', alpha=0.5)

    # Set the plot title and axis labels
    plt.title('Scatter Plot of 1114 Points')
    plt.xlabel('demographic points')
    plt.ylabel('cohen_kappa_scores')
    plt.savefig('demo_to_agreement_score.png', format='png',
                dpi=300)  # Specify the file name, format, and dpi (optional)

    # Show the plot
    plt.show()


def visualize_survey_to_agreement_score(survey_list, survey_score_list):
    # Generate 15 data points
    data_points = np.array(survey_score_list)

    # Generate y-axis labels
    y_labels = [survey_name_map[survey] for survey in survey_list]
    # y_labels = survey_list

    # Create a horizontal bar chart
    plt.barh(y_labels, data_points, color='blue')

    # Set the plot title and axis labels
    plt.title(f'Horizontal Bar Chart of {len(survey_list)} Data Points')
    plt.xlabel('cohen_kappa_scores')
    plt.ylabel('survey name')

    # Adjust the left margin of the plot
    plt.subplots_adjust(left=0.4)  # Increase the left margin to 0.2 inches
    plt.savefig('survey_to_agreement_score.png', format='png',
                dpi=300)  # Specify the file name, format, and dpi (optional)
    # Show the plot
    plt.show()

def visualize_topic_to_agreement_score(topic_list, topic_score_list):
    # Generate 15 data points
    data_points = np.array(topic_score_list)

    # Generate y-axis labels
    y_labels = topic_list

    plt.figure(figsize=(20, 20))
    # Create a horizontal bar chart
    plt.barh(y_labels, data_points, color='blue')

    # Set the plot title and axis labels
    plt.title(f'Horizontal Bar Chart of {len(topic_list)} Data Points')
    plt.xlabel('cohen_kappa_scores')
    plt.ylabel('survey name')
    # plt.yticks(rotation=30)

    # Adjust the left margin of the plot
    plt.subplots_adjust(left=0.4)  # Increase the left margin to 0.2 inches
    plt.savefig('topic_to_agreement_score.png', format='png',
                dpi=300)  # Specify the file name, format, and dpi (optional)
    # Show the plot
    plt.show()


def find_pair_by_demographic(all_user_responses):

    demographic_dict = {}
    for user_resp in all_user_responses:
        user_id = user_resp['user_id']
        survey_name = user_resp['survey_name']
        implicit_info_dict = user_resp['implicit_info']
        explicit_info_dict = user_resp['explicit_info']
        demo_key = str(tuple(sorted(explicit_info_dict.items())))

        if demo_key not in demographic_dict.keys():
            demographic_dict[demo_key] = {}

        ##### construct demographic_dict #####
        for q_id, implicit_dict in implicit_info_dict.items():
            question = implicit_dict['question']
            choice = implicit_dict['choice']
            answer = implicit_dict['answer']
            subtopic_fg = implicit_dict['subtopic_fg']
            subtopic_cg = implicit_dict['subtopic_cg']
            topic_mapping_key = get_topic_mapping_key(question, choice)

            if q_id in demographic_dict[demo_key].keys():
                demographic_dict[demo_key][q_id]['answers'].append((user_id, answer))
            else:
                demographic_dict[demo_key][q_id] = {
                    "answers": [(user_id, answer)],
                    "choice": choice,
                    "subtopic_fg": subtopic_fg,
                    "subtopic_cg": subtopic_cg,
                    "topic_mapping_key": topic_mapping_key,
                    "survey_name": survey_name,
                }
                # print("demographic_dict", demographic_dict)

    # select demographics & qa that have more than two answers
    demographic_pair_dict = {}
    for demo_key in demographic_dict.keys():
        qa_dict = demographic_dict[demo_key]
        for q_id, q_info in qa_dict.items():
            if len(q_info["answers"]) > 1:
                if demo_key not in demographic_pair_dict.keys():
                    demographic_pair_dict[demo_key] = {}
                demographic_pair_dict[demo_key][q_id] = q_info

    with open("demographic_pair_dict.json", "w") as f:
        json.dump(demographic_pair_dict, f, indent=4)
    print(len(demographic_dict), len(demographic_pair_dict))

    return demographic_pair_dict


def calculate_cohen_kappa_score(demographic_pair_dict):
    print()

    demo_to_agreement_score = {}
    survey_to_agreement_score = {}

    for demo_key, qa_dict in demographic_pair_dict.items():
        ### get all user ids ###
        all_user_ids = set()
        for q_id, response in qa_dict.items():
            answers = response['answers']
            user_ids = [ans[0] for ans in answers]
            all_user_ids.update(user_ids)
        ### end ###

        all_user_id_to_response = {}
        all_user_id_to_survey = {}
        all_user_id_to_qid = {}
        for user_id in list(all_user_ids):
            all_user_id_to_response[user_id] = []
            all_user_id_to_survey[user_id] = []
            all_user_id_to_qid[user_id] = []

        for q_id, response in qa_dict.items():
            answers = response['answers']
            survey_name = response['survey_name']
            user_id_ans_dict = {}
            for ans in answers:
                # ans[0]: user_id, ans[1]: user_response
                user_id_ans_dict[ans[0]] = {
                    "response": ans[1],
                    "survey_name": survey_name,
                    "question_id": q_id,
                } # {8307: {'response': 'No', 'survey_name': 'American_Trends_Panel_W29', 'question_id': 'PERSDISCR_W29'}}

            # print("user_id_ans_dict:", user_id_ans_dict)
            # construct user response dict
            # {
            #   user_id1: [resp1, resp2, ...],
            #   user_id2: [resp1, resp2, ...],
            # }
            for user_id in list(all_user_ids):
                if user_id in user_id_ans_dict.keys():
                    all_user_id_to_response[user_id].append(user_id_ans_dict[user_id]['response'])
                    all_user_id_to_survey[user_id].append(user_id_ans_dict[user_id]['survey_name'])
                    all_user_id_to_qid[user_id].append(user_id_ans_dict[user_id]['question_id'])
                else:
                    all_user_id_to_response[user_id].append(None)
                    all_user_id_to_survey[user_id].append(None)
                    all_user_id_to_qid[user_id].append(None)

        ##### calculate cohen kappa scores for all user combinations #####
        all_user_ids = itertools.combinations(list(all_user_id_to_response.keys()), 2)
        for user_id_comb in all_user_ids:
            user_id1, user_id2 = user_id_comb
            resp1 = all_user_id_to_response[user_id1]
            resp2 = all_user_id_to_response[user_id2]

            surv_name1 = all_user_id_to_survey[user_id1]
            surv_name2 = all_user_id_to_survey[user_id2]

            qid1 = all_user_id_to_qid[user_id1]
            qid2 = all_user_id_to_qid[user_id2]

            assert len(resp1) == len(resp2)

            inter_index = [i for i in range(len(resp1)) if resp1[i] is not None and resp2[i] is not None]
            if len(inter_index) == 0:
                continue

            extracted_resp1, extracted_resp2 = [], []
            extracted_surv1, extracted_surv2 = set(), set()
            extracted_qid1, extracted_qid2 = [], []
            for idx in inter_index:
                extracted_resp1.append(resp1[idx])
                extracted_resp2.append(resp2[idx])
                extracted_surv1.add(surv_name1[idx])
                extracted_surv2.add(surv_name2[idx])
                extracted_qid1.append(qid1[idx])
                extracted_qid2.append(qid2[idx])

            assert len(extracted_surv1) == len(extracted_surv2) == 1
            assert len(extracted_qid1) == len(extracted_qid2)

            agreement_score = cohen_kappa_score(extracted_resp1, extracted_resp2)
            # print(user_id1, user_id2, "inter_index:", len(inter_index), inter_index)
            # print("extracted_resp1:", len(extracted_resp1), extracted_resp1)
            # print("extracted_resp2:", len(extracted_resp2), extracted_resp2)
            # print("extracted_surv1:", len(extracted_surv1), extracted_surv1, "extracted_surv2:", len(extracted_surv2), extracted_surv2)
            # print("extracted_qid1:", len(extracted_qid1), extracted_qid1)
            # # print("extracted_qid2:", len(extracted_qid2), extracted_qid2)
            #
            # print("agreement_score:", agreement_score)
            # print()

            if demo_key not in demo_to_agreement_score.keys():
                demo_to_agreement_score[demo_key] = []

            survey_name = list(extracted_surv1)[0]
            if survey_name not in survey_to_agreement_score.keys():
                survey_to_agreement_score[survey_name] = []

            demo_to_agreement_score[demo_key].append(agreement_score)
            survey_to_agreement_score[survey_name].append(agreement_score)


    # visualize agreement score per each demographic
    demo_key_list, demo_score_list = [], []
    for demo_key, scores in demo_to_agreement_score.items():
        demo_key_list.append(demo_key)
        demo_score_list.append(sum(scores)/len(scores))

    # visualize agreement score per each topic W##
    survey_list, survey_score_list = [], []
    for survey_name, scores in survey_to_agreement_score.items():
        survey_list.append(survey_name)
        survey_score_list.append(sum(scores) / len(scores))

    visualize_demo_to_agreement_score(demo_key_list, demo_score_list)
    visualize_survey_to_agreement_score(survey_list, survey_score_list)



def find_pair_by_topic(all_user_responses):
    """
    {subtopic_cg:
        q_id: {
            "answers": [(user_id, answer)],
            "choice": choice,
            "topic_mapping_key": topic_mapping_key,
            "survey_name": survey_name,
        }
    }
    :param all_user_responses:
    :return:
    """
    topic_dict = {}
    for user_resp in all_user_responses:
        user_id = user_resp['user_id']
        survey_name = user_resp['survey_name']
        implicit_info_dict = user_resp['implicit_info']
        explicit_info_dict = user_resp['explicit_info']
        demo_key = str(tuple(sorted(explicit_info_dict.items())))

        ##### construct topic_dict #####
        for q_id, implicit_dict in implicit_info_dict.items():
            question = implicit_dict['question']
            choice = implicit_dict['choice']
            answer = implicit_dict['answer']
            subtopic_fg = implicit_dict['subtopic_fg']
            subtopic_cg = str(tuple(implicit_dict['subtopic_cg']))
            # topic_mapping_key = get_topic_mapping_key(question, choice)

            if subtopic_cg not in topic_dict.keys():
                topic_dict[subtopic_cg] = {}

            if demo_key not in topic_dict[subtopic_cg].keys():
                topic_dict[subtopic_cg][demo_key] = {}

            if q_id not in topic_dict[subtopic_cg][demo_key].keys():
                topic_dict[subtopic_cg][demo_key][q_id] = {
                    "answers": [(user_id, answer)],
                    "choice": choice,
                    "survey_name": survey_name,
                }
            else:
                topic_dict[subtopic_cg][demo_key][q_id]["answers"].append(((user_id, answer)))

    # with open("topic_dict.json", "w") as f:
    #     json.dump(topic_dict, f, indent=4)

    # select topic & qa that have more than two answers
    topic_pair_dict = {}
    for topic in topic_dict.keys():
        demo_to_qa = topic_dict[topic]
        for demo_key, qa_info in demo_to_qa.items():
            for q_id, q_info in qa_info.items():
                if len(q_info["answers"]) > 1:
                    if topic not in topic_pair_dict.keys():
                        topic_pair_dict[topic] = {}

                    if demo_key not in topic_pair_dict[topic].keys():
                        topic_pair_dict[topic][demo_key] = {}

                    topic_pair_dict[topic][demo_key][q_id] = q_info

    with open("topic_pair_dict.json", "w") as f:
        json.dump(topic_pair_dict, f, indent=4)


def calculate_cohen_kappa_score_per_subtopic(topic_pair_dict):
    topic_to_agreement_score = {}
    for topic, demo_to_qa in topic_pair_dict.items():
        for demo_key, qa_info in demo_to_qa.items():
            ### get all user ids ###
            all_user_ids = set()
            for q_id, response in qa_info.items():
                answers = response['answers']
                user_ids = [ans[0] for ans in answers]
                all_user_ids.update(user_ids)
            ### end ###

            # print("demo_key:", demo_key)
            # print("all_user_ids:", all_user_ids)

            all_user_id_to_response = {}
            for user_id in list(all_user_ids):
                all_user_id_to_response[user_id] = []

            for q_id, response in qa_info.items():
                answers = response['answers']
                user_id_ans_dict = {}
                for ans in answers:
                    # ans[0]: user_id, ans[1]: user_response
                    user_id_ans_dict[ans[0]] = {
                        "response": ans[1]
                    }

                for user_id in list(all_user_ids):
                    if user_id in user_id_ans_dict.keys():
                        all_user_id_to_response[user_id].append(user_id_ans_dict[user_id]['response'])
                    else:
                        all_user_id_to_response[user_id].append(None)

            ##### calculate cohen kappa scores for all user combinations #####
            all_user_ids = itertools.combinations(list(all_user_id_to_response.keys()), 2)
            for user_id_comb in all_user_ids:
                user_id1, user_id2 = user_id_comb
                resp1 = all_user_id_to_response[user_id1]
                resp2 = all_user_id_to_response[user_id2]

                assert len(resp1) == len(resp2)

                inter_index = [i for i in range(len(resp1)) if resp1[i] is not None and resp2[i] is not None]
                if len(inter_index) == 0:
                    continue

                extracted_resp1, extracted_resp2 = [], []
                is_disagree = False
                for idx in inter_index:
                    if resp1[idx] != resp2[idx]: # to prevent nan issue when calculating agreement score
                        is_disagree = True
                    extracted_resp1.append(resp1[idx])
                    extracted_resp2.append(resp2[idx])

                assert len(extracted_resp1) == len(extracted_resp2)

                if is_disagree:
                    agreement_score = cohen_kappa_score(extracted_resp1, extracted_resp2)
                else:
                    agreement_score = 1
                # print("agreement_score:", agreement_score)

            if topic not in topic_to_agreement_score.keys():
                topic_to_agreement_score[topic] = {}

            if demo_key not in topic_to_agreement_score[topic].keys():
                topic_to_agreement_score[topic][demo_key] = []

            topic_to_agreement_score[topic][demo_key].append(agreement_score)

    topic_scores = {}
    for topic, demo_to_scores in topic_to_agreement_score.items():
        if topic not in topic_scores.keys():
            topic_scores[topic] = []

        for demo_key, scores in demo_to_scores.items():
            topic_scores[topic].extend(scores)

    topic_list, topic_score_list = [], []
    for topic, scores in topic_scores.items():
        topic_list.append(topic)
        topic_score_list.append(sum(scores)/len(scores))

    visualize_topic_to_agreement_score(topic_list, topic_score_list)

def create_demo_to_qa_dict(resp_indi_dict, all_qinfo_dict):
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

    # print("demo_to_question_dict:", demo_to_question_dict)
    new_demo_to_question_dict = {}
    demo_qa_num = []
    for key, qa_pair in demo_to_question_dict.items():
        demo_key = key
        demo_q_list = []
        for q, a in qa_pair.items():
            if len(a) <= 1: # or len(a) > 2:
                continue
            print("question:", q)
            print("answers:", a)
            assert False
            demo_q_list.append({q: a})
        if len(demo_q_list) > 0:
            new_demo_to_question_dict[demo_key] = demo_q_list
            demo_qa_num.append(len(demo_q_list))

    print("avg_demo_qa_num:", sum(demo_qa_num) / len(demo_qa_num))

    demo_to_question_dict = {}
    for i, (demo_key, demo_q_list) in enumerate(new_demo_to_question_dict.items()):
        agreement = 0
        disagreement = 0
        new_demo_q_list = []
        for demo_q in demo_q_list:
            for k, v in demo_q.items():
                answers = v
                for combi in list(itertools.combinations(answers, 2)):
                    if combi[0] == combi[1]:
                        agreement += 1
                    else:
                        disagreement += 1
                question = all_qinfo_dict[k]['question']
                choices = all_qinfo_dict[k]['choice']
                choices = ast.literal_eval(choices)
                choices = "/".join(choices)
                new_demo_q_list.append({
                    f"{question} [{choices}]": answers
                })
        demo_to_question_dict[str(demo_key)] = {
            "qa_pairs": new_demo_q_list,
            "agreement": agreement,
            "disagreement": disagreement
        }
        print(i, "demo_q_list:", len(demo_q_list), "agreement:", agreement, "disagreement:", disagreement)


    print("new_demo_to_question_dict:", len(new_demo_to_question_dict))
    print("total_keys:", len(resp_indi_dict.keys()))
    # print("total_responses:", total_responses)
    # print("total_pairs:", len(pairs))


    return demo_to_question_dict


def aggregate_responses_by_topic(all_user_responses):
    print()
    all_resp_by_subtopic = {}
    all_resp_by_topic = {}
    for resp in all_user_responses:
        user_id = resp["user_id"]
        survey_name = resp["survey_name"]
        implicit_info = resp["implicit_info"]

        for q_id, q_info in implicit_info.items():
            question = q_info['question']
            answer = q_info['answer']
            subtopic_cg = q_info['subtopic_cg']

            # subtopic_key = str((survey_name, subtopic_cg))
            ##### main topic #####
            topic_key = survey_name
            if topic_key not in all_resp_by_topic.keys():
                all_resp_by_topic[topic_key] = {}

            if user_id not in all_resp_by_topic[topic_key].keys():
                all_resp_by_topic[topic_key][user_id] = []

            all_resp_by_topic[topic_key][user_id].append((q_id, answer))
            ########################

            ##### main topic #####
            for subtopic in subtopic_cg:
                if subtopic not in all_resp_by_subtopic.keys():
                    all_resp_by_subtopic[subtopic] = {}

                if user_id not in all_resp_by_subtopic[subtopic].keys():
                    all_resp_by_subtopic[subtopic][user_id] = []

                all_resp_by_subtopic[subtopic][user_id].append((q_id, answer))
            ########################


    # print("all_resp_by_subtopic:", len(all_resp_by_subtopic))
    # print("all_resp_by_topic:", len(all_resp_by_topic), all_resp_by_topic)

    # with open("all_resp_by_subtopic.json", "w") as f:
    #     json.dump(all_resp_by_subtopic, f, indent=4)
    #
    # with open("all_resp_by_topic.json", "w") as f:
    #     json.dump(all_resp_by_topic, f, indent=4)


def split_data(json_data_path, train_survey_to_resp, test_survey_to_resp):
    with open(json_data_path) as fd:
        resp_by_topic = json.load(fd)

    train_user_ids, test_user_ids = {}, {}
    for topic, user_responses in train_survey_to_resp.items():
        for user_resp in user_responses:
            user_id = user_resp["user_id"]
            if topic not in train_user_ids.keys():
                train_user_ids[topic] = []
            train_user_ids[topic].append(user_id)

    for topic, user_responses in test_survey_to_resp.items():
        for user_resp in user_responses:
            user_id = user_resp["user_id"]
            if topic not in test_user_ids.keys():
                test_user_ids[topic] = []
            test_user_ids[topic].append(user_id)

    train_checklist_dict, train_eval_dict = {}, {}
    test_checklist_dict, test_eval_dict = {}, {}
    count = 0
    for topic, user_responses in resp_by_topic.items():
        for user_id, qa_pair in user_responses.items():
            if len(qa_pair) == 1:
                continue
            user_id = int(user_id)

            if user_id in random.sample(train_user_ids[topic], 2) and count <= 20:
                x_val, x_test = train_test_split(qa_pair, random_state=42, test_size=0.8)
                if topic not in train_checklist_dict.keys():
                    train_checklist_dict[topic] = {}

                if topic not in train_eval_dict.keys():
                    train_eval_dict[topic] = {}

                if user_id not in train_checklist_dict[topic].keys():
                    train_checklist_dict[topic][user_id] = []

                if user_id not in train_eval_dict[topic].keys():
                    train_eval_dict[topic][user_id] = []

                train_checklist_dict[topic][user_id] = x_val
                train_eval_dict[topic][user_id] = x_test
                count += 1

            if user_id in test_user_ids[topic]:
                x_val, x_test = train_test_split(qa_pair, random_state=42, test_size=0.8)
                if topic not in test_checklist_dict.keys():
                    test_checklist_dict[topic] = {}

                if topic not in test_eval_dict.keys():
                    test_eval_dict[topic] = {}

                if user_id not in test_checklist_dict[topic].keys():
                    test_checklist_dict[topic][user_id] = []

                if user_id not in test_eval_dict[topic].keys():
                    test_eval_dict[topic][user_id] = []

                test_checklist_dict[topic][user_id] = x_val
                test_eval_dict[topic][user_id] = x_test

    print(len(train_checklist_dict), len(train_eval_dict), len(test_checklist_dict), len(test_eval_dict))

    return train_checklist_dict, train_eval_dict, test_checklist_dict, test_eval_dict