import argparse
import ast
import datetime
import json
import logging
import math
import random
from pathlib import Path
import time
import openai


def set_logger(args):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')
    log_dir = "./log/"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging_path = log_dir + f"output_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logging_path),
            logging.StreamHandler()
        ]
    )


def generate_output_with_implicit(args, val_dict, test_dict, all_qinfo_dict, all_demographic_dict, api_key):
    set_logger(None)
    logger = logging.getLogger()

    all_topics = sorted(list(val_dict.keys()))
    if args.num_topics == -1:
        num_topics = len(all_topics)
    else:
        num_topics = args.num_topics

    if args.num_users != -1:
        num_users = args.num_users

    if args.num_test != -1:
        num_test = args.num_test

    if args.num_checklist != -1:
        num_checklist = args.num_checklist

    for topic in all_topics[:num_topics]:

        if args.num_users == -1:
            user_ids_per_topic = sorted(list(val_dict[topic].keys()))
        else:
            user_ids_per_topic = sorted(list(val_dict[topic].keys()))[:num_users]

        for user_id in user_ids_per_topic:
            logger.info("======================================== user id: {} ===========================================".format(user_id))
            val_responses = val_dict[topic][user_id]
            test_responses = test_dict[topic][user_id]

            checklist = []
            output_mapping = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for val_resp in val_responses:
                q_id, answer = val_resp
                question = all_qinfo_dict[q_id]["question"].strip()
                choices = ast.literal_eval(all_qinfo_dict[q_id]["choice"])
                choices = [choice for choice in choices] # to remove single quote
                choice_idx = choices.index(answer)

                choices = [f"{output_mapping[i]}. {choice}"for i, choice in enumerate(choices)]
                choice = " ".join(choices)

                prompt = f"There is a question: \"{question}\" with the following answer choices: {choice}. This person chose {output_mapping[choice_idx]}.\n"
                checklist.append(prompt)

            # Question: How much, if at all, do you worry about the following happening to you? Being the victim of a terrorist attack\nA.Worry a lot\nB.Worry a little\nC.Do not worry at all\nD.Refused\nAnswer:
            test_questions = []
            test_answers = []
            test_choices = []
            if num_test < len(test_responses):
                sampled_test_responses = random.sample(test_responses, num_test)
            else:
                sampled_test_responses = test_responses

            for test_resp in sampled_test_responses:
                q_id, answer = test_resp
                question = all_qinfo_dict[q_id]["question"].strip()
                choices = ast.literal_eval(all_qinfo_dict[q_id]["choice"])
                choices = [choice for choice in choices] # to remove single quote
                test_choices.append(choices)

                choices = [f"{output_mapping[i]}.{choice}\n" for i, choice in enumerate(choices)]
                choice = "".join(choices)

                prompt = f"Question: {question}\n{choice}Answer:"
                test_questions.append(prompt)
                test_answers.append(answer)

            # print("checklist", len(checklist), checklist[:5])
            # print("test_questions", len(test_questions), test_questions[:5])
            # print("test_questions", len(test_questions), test_answers[:5])
            openai.api_key = api_key
            generated_output = []
            for question, answer, choices in zip(test_questions, test_answers, test_choices):
                prior = "\n".join(checklist[:num_checklist])
                prompt = prior + f"How would this person answer the following question?\n" + question
                # prompt = prior + question

                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    temperature=0,
                    max_tokens=1,
                    top_p=1,
                    n=1
                )

                response_choice = response["choices"][0]["text"].strip()
                generated_output.append(response_choice)
                logger.info("================================================================================================")
                logger.info("prompt: {}".format(prompt))
                logger.info("choices: {}".format(choices))
                logger.info("user_answer: {}".format(answer))
                logger.info("model_choice: {}".format(response_choice))
                logger.info("open ai response: {}".format(response))
                logger.info("================================================================================================")


            assert len(test_answers) == len(generated_output)
            correct = 0
            incorrect = 0
            for choices, user_answer, model_choice in zip(test_choices, test_answers, generated_output):
                choice_idx = choices.index(user_answer)
                user_choice = output_mapping[choice_idx]
                if user_choice == model_choice:
                    correct += 1
                else:
                    incorrect += 1

            logger.info("correct instances: {}, incorrect instances: {}, total instances: {}".format(correct, incorrect, len(test_answers)))
        logger.info("======================================== done user id: {} ===========================================".format(user_id))



def generate_output_with_explicit(args, val_dict, test_dict, all_qinfo_dict, all_demographic_dict, api_key):
    set_logger(None)
    logger = logging.getLogger()

    all_topics = sorted(list(val_dict.keys()))

    if args.num_topics == -1:
        num_topics = len(all_topics)
    else:
        num_topics = args.num_topics

    if args.num_users != -1:
        num_users = args.num_users

    if args.num_test != -1:
        num_test = args.num_test

    demo_map = {
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

    # topic = all_topics[0]
    for topic in all_topics[:num_topics]:
        if args.num_users == -1:
            user_ids_per_topic = sorted(list(val_dict[topic].keys()))
        else:
            user_ids_per_topic = sorted(list(val_dict[topic].keys()))[:num_users]
        for user_id in user_ids_per_topic:
            logger.info("======================================== user id: {} ===========================================".format(user_id))
            demographic_info = all_demographic_dict[user_id]
            # val_responses = val_dict[topic][user_id]
            test_responses = test_dict[topic][user_id]

            checklist = []
            for key, value in demographic_info.items():
                if isinstance(value, float) and math.isnan(value):
                    continue
                field = demo_map[key]
                demo_info = f"{field}: {value}"
                checklist.append(demo_info)
            checklist = "\n".join(checklist)

            output_mapping = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            test_questions = []
            test_answers = []
            test_choices = []

            if num_test < len(test_responses):
                sampled_test_responses = random.sample(test_responses, num_test)
            else:
                sampled_test_responses = test_responses

            for test_resp in sampled_test_responses:
                q_id, answer = test_resp
                question = all_qinfo_dict[q_id]["question"].strip()
                choices = ast.literal_eval(all_qinfo_dict[q_id]["choice"])
                choices = [choice for choice in choices] # to remove single quote
                test_choices.append(choices)

                choices = [f"{output_mapping[i]}.{choice}\n" for i, choice in enumerate(choices)]
                choice = "".join(choices)

                prompt = f"Question: {question}\n{choice}Answer:"
                test_questions.append(prompt)
                test_answers.append(answer)

            # print("test_questions", len(test_questions), test_questions[:5])
            # print("test_questions", len(test_questions), test_answers[:5])
            openai.api_key = api_key
            generated_output = []
            for question, answer, choices in zip(test_questions, test_answers, test_choices):
                prior = checklist
                prompt = "This person can be described as follows:\n" + prior + \
                         "\nHow would this person answer the following question?\n" + question
                # prompt = prior + question

                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    temperature=0,
                    max_tokens=1,
                    top_p=1,
                    n=1
                )

                response_choice = response["choices"][0]["text"].strip()
                generated_output.append(response_choice)
                logger.info("================================================================================================")
                logger.info("prompt: {}".format(prompt))
                logger.info("choices: {}".format(choices))
                logger.info("user_answer: {}".format(answer))
                logger.info("model_choice: {}".format(response_choice))
                logger.info("open ai response: {}".format(response))
                logger.info("================================================================================================")


            assert len(test_answers) == len(generated_output)
            correct = 0
            incorrect = 0
            for choices, user_answer, model_choice in zip(test_choices, test_answers, generated_output):
                choice_idx = choices.index(user_answer)
                user_choice = output_mapping[choice_idx]
                if user_choice == model_choice:
                    correct += 1
                else:
                    incorrect += 1

            logger.info("correct instances: {}, incorrect instances: {}, total instances: {}".format(correct, incorrect, len(test_answers)))
        logger.info("======================================== done user id: {} ===========================================".format(user_id))


def main(args):
    all_qinfo_file = "all_qinfo_dict.json"
    all_demographic_file = "all_demographic_dict.json"
    with open(all_qinfo_file, "r") as fd:
        all_qinfo_dict = json.load(fd)
        print("all_qinfo_dict", len(all_qinfo_dict))
    with open(all_demographic_file, "r") as fd:
        all_demographic_dict = json.load(fd)
        print("all_demographic_dict", len(all_demographic_dict))

    train_checklist_dict_file = "sample_train_checklist_dict.json"
    train_eval_dict_file = "sample_train_eval_dict.json"
    with open(train_checklist_dict_file, "r") as fd:
        train_checklist_dict = json.load(fd)
        print("train_checklist_dict", len(train_checklist_dict))
    with open(train_eval_dict_file, "r") as fd:
        train_eval_dict_file = json.load(fd)
        print("train_eval_dict_file", len(train_eval_dict_file))

    # test_checklist_dict_file = "test_checklist_dict.json"
    # test_eval_dict_file = "test_eval_dict.json"
    # with open(test_checklist_dict_file, "r") as fd:
    #     test_checklist_dict = json.load(fd)
    #     print("test_checklist_dict", len(test_checklist_dict))
    # with open(test_eval_dict_file, "r") as fd:
    #     test_eval_dict = json.load(fd)
    #     print("test_eval_dict", len(test_eval_dict))

    openai_key = args.openai_key
    generate_output_with_implicit(args, train_checklist_dict, train_eval_dict_file, all_qinfo_dict, all_demographic_dict, openai_key)
    generate_output_with_explicit(args, train_checklist_dict, train_eval_dict_file, all_qinfo_dict, all_demographic_dict, openai_key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_key", type=str, default="", help="open api key")
    parser.add_argument("--num_topics", type=int, default=1, help="# of topics")
    parser.add_argument("--num_users", type=int, default=3, help="# of users")
    parser.add_argument("--num_checklist", type=int, default=1, help="# of checklist")
    parser.add_argument("--num_test", type=int, default=20, help="# of test prompts")
    args = parser.parse_args()
    main(args)
