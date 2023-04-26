import argparse
import ast
import json
import os.path
from pathlib import Path
from typing import List, Dict

import pandas as pd
from gptinference import utils
from tqdm import tqdm

from gptinference.base_prompt import Prompt
from gptinference.openai_wrapper import OpenAIWrapper
from gptinference.utils import read_jsonl_or_json, write_json

from synthesize_opinionqa.utils import DEMO_MAP

OUTPUT_MAP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PEW_SURVEY_LIST = [26, 27, 29, 32, 34, 36, 41, 42, 43, 45, 49, 50, 54, 82, 92]

class PersonaCreator(Prompt):
    def __init__(self, engine: str, openai_wrapper: OpenAIWrapper, demo_metadata: List[Dict] or Dict):
        super().__init__()
        self.openai_wrapper = openai_wrapper
        self.engine = engine
        self.demo_metadata = demo_metadata

    def make_query(self, implicit_persona: List[str], topic: str) -> str:
        if not implicit_persona or not topic:
            return ""

        # implicit opinions
        implicit_persona_list = []
        for persona in implicit_persona:
            implicit_persona_list.append(persona["declarative_opinion"])
        implicit_persona_list = [f"{i+1}. {persona}\n" for i, persona in enumerate(implicit_persona_list)]
        implicit_persona_str = "".join(implicit_persona_list)

        metadata_str = []
        for metadata in demo_metadata:
            for demo_key, options in metadata.items():
                options_str = [f"{OUTPUT_MAP[i]}.{option}" for i, option in enumerate(options)]
                options_str = "\n".join(options_str)
                # print(f"{demo_key}:\n{options_str}\n")
                metadata_str.append(f"{demo_key}:\n{options_str}\n")
        metadata_str = "\n".join(metadata_str)

        prompt = \
f"""A person has the following opinions on {topic}.

Opinions:
{implicit_persona_str}
Based on the above list of opinions, pick one choice for each demographic category to describe this person:

{metadata_str}
Answer:
"""
        return prompt

    def __call__(self, implicit_persona: List[str], topic:str) -> str:
        generation_query = self.make_query(
            implicit_persona=implicit_persona,
            topic=topic,
        )

        generated_sent = self.openai_wrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=200,
            stop_token="###",
            temperature=0.0
        )

        return generated_sent.strip()  # gpt3 turbo adds newline in the beginning so strip it.


def early_stopping(topicwise_ctr, topic, max_topics, max_users):
    cond_users  = max_users > 0 and topicwise_ctr.get(topic, 0) >  max_users
    if cond_users:
        return True

    cond_topics = max_topics > 0 and len(topicwise_ctr) > max_topics
    if cond_topics:
        return True

    return False

class PersonalizedQA:
    """
    save personalized opinion to output json.
    """
    def __init__(self, args):
        self.num_implicit = args.num_implicit

    def personalized_qa(self, persona: PersonaCreator, user_responses_jsons: Dict, max_users:int, max_topics: int):
        """ get demographic answer choices based on implicit persona.
        """
        model_generated = []
        topicwise_ctr = dict()
        for user_response_json in tqdm(user_responses_jsons, desc="users..."):

            user_id = user_response_json["user_id"]
            topic = user_response_json["topic"]
            if topic not in topicwise_ctr:
                topicwise_ctr[topic] = 0
            topicwise_ctr[topic] += 1

            if early_stopping(topicwise_ctr=topicwise_ctr, max_topics=max_topics, max_users=max_users, topic=topic):
                continue

            if self.num_implicit > len(user_response_json["implicit_persona"]):
                num_implicit = len(user_response_json["implicit_persona"])
            else:
                num_implicit = self.num_implicit
            implicit_persona = user_response_json["implicit_persona"][:num_implicit]
            explicit_persona = user_response_json["explicit_persona"]

            try:
                model_choice = persona(
                    implicit_persona=implicit_persona,
                    topic=topic,
                )
                model_generated.append({
                    "user_id": user_id,
                    "topic": topic,
                    "model_choice": model_choice,
                    "user_choice": explicit_persona,
                })
            except Exception as exc:
                model_generated.append({
                    "user_id": user_id,
                    "topic": topic,
                    "model_choice": "UNKNOWN",
                    "user_choice": explicit_persona,
                })
                print(f"{exc}")

        return model_generated


def calculate_accuracy(model_generation_path):
    model_generation = read_jsonl_or_json(model_generation_path)
    print("================ model_generation_path: {} ================".format(model_generation_path))
    accuracy_list = []
    user_accuracy_list = []
    user_choice_list = []
    for model_output in model_generation:
        user_id = model_output["user_id"]
        topic = model_output["topic"]
        model_choice = model_output["model_choice"]
        user_choice = model_output["user_choice"]

        # 'Region: D. West', 'Age: A. 18-29', 'Gender: B. Female',
        model_choice = model_choice.split("\n")

        # parse model_choice
        model_choice_dict = dict()
        for choice in model_choice:
            choice = choice.split(":")
            demo_key = choice[0].strip()
            demo_ans = choice[-1].strip()
            model_choice_dict[demo_key] = demo_ans

        user_choice_dict = dict()
        for choice in user_choice:
            for demo_key, value in choice.items():
                if value.lower() == "refused":
                    continue
                user_choice_dict[demo_key] = value

        total_num = len(user_choice_dict)
        correct, incorrect = 0, 0
        correct_key, incorrect_key = [], []
        for demo_key, user_value in user_choice_dict.items():
            if demo_key not in model_choice_dict:
                incorrect_key.append(demo_key)
                incorrect += 1
                continue

            if user_value in model_choice_dict[demo_key]:
                correct_key.append(demo_key)
                correct += 1
            else:
                incorrect_key.append(demo_key)
                incorrect += 1
        accuracy_per_user = correct / total_num
        user_accuracy_list.append({user_id: accuracy_per_user})
        user_choice_list.append({
            user_id: {
                "correct": correct_key,
                "incorrect": incorrect_key
            }
        })
        accuracy_list.append(accuracy_per_user)

    final_accuracy = sum(accuracy_list) / len(accuracy_list)
    return {"accuracy": final_accuracy, "user-accuracy": user_accuracy_list, "answer-choices": user_choice_list}

def load_metadata(metadata_path):
    meta_df = pd.read_csv(metadata_path)
    meta_dict = meta_df.to_dict()
    key_dict = meta_dict['key']
    options_dict = meta_dict['options']

    demo_metadata = []
    for key, key_name in sorted(key_dict.items()):
        demo_name = DEMO_MAP[key_name]
        options = ast.literal_eval(options_dict[key])
        options = [opt for opt in options]
        demo_metadata.append({demo_name: options})

    return demo_metadata

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="data/opinionqa/sampled_user_responses_decl.json", help="json path")
    parser.add_argument("--out_dir", type=str, default="data/model-output/", help="json path")
    parser.add_argument("--cache_path", type=str, default="data/cache/gpt_demo_cache.jsonl", help="json path")
    parser.add_argument("--metadata_path", type=str, default="data/opinion-qa/human_resp/American_Trends_Panel_W26/metadata.csv", help="metadata choices json file")
    parser.add_argument("--num_implicit", type=int, default=2, help="number of implicit persona to use")
    parser.add_argument("--max_users", type=int, default=4, help="max num users to do inference on.")
    parser.add_argument("--max_topics", type=int, default=3, help="max topics to test inference on (currently, ~15).")
    parser.add_argument("--max_retries", type=int, default=2, help="max number of openai retries when a call fails.")
    # parser.add_argument("--max_ques", type=int, default=30, help="max ques to test inference on.")
    # parser.add_argument("--option", type=int, default=0, choices=[-1, 0, 1, 2], help="-1: no-persona, 0: implicit, 1: explicit, 2: both")

    args = parser.parse_args()
    os.environ["OPENAI_MAX_TRIES_INT"] = str(args.max_retries)

    dir_name = f"demo_{args.num_implicit}pts-t{args.max_topics}-u{args.max_users}"
    print(f"\nStart experiment: {dir_name} ...")

    demo_metadata = load_metadata(args.metadata_path) # "../data/opinion-qa/human_resp/American_Trends_Panel_W26/metadata.csv"

    persona = PersonaCreator(engine="text-davinci-003", openai_wrapper=OpenAIWrapper(cache_path=args.cache_path),
                             demo_metadata=demo_metadata)
    output = PersonalizedQA(args).personalized_qa(
        persona=persona,
        user_responses_jsons=read_jsonl_or_json(args.in_path),
        max_users=args.max_users,
        max_topics=args.max_topics
    )

    output_dir = os.path.join(args.out_dir, dir_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, "model_generation.json")
    write_json(outpath=output_file, json_data=output)

    metrics = calculate_accuracy(output_file)
    metrics_file = os.path.join(output_dir, "model_accuracy.json")
    write_json(outpath=metrics_file, json_data=metrics)