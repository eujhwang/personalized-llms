import argparse
import json
import os.path
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from gptinference.base_prompt import Prompt
from gptinference.openai_wrapper import OpenAIWrapper
from gptinference.utils import read_jsonl_or_json, write_json

OUTPUT_MAP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class PersonaCreator(Prompt):
    def __init__(self, engine: str, openai_wrapper: OpenAIWrapper):
        super().__init__()
        self.openai_wrapper = openai_wrapper
        self.engine = engine

    def make_query(self, implicit_persona: List[str], explicit_persona: List[str],
                   topic: str, question: List[str], choices: List[str]) -> str:
        if not question or not choices:
            return ""

        # implicit prompt
        if implicit_persona:
            implicit_persona_list = []
            for persona in implicit_persona:
                implicit_persona_list.append(persona["declarative_opinion"])
            implicit_persona_list = [f"{i+1}. {persona}\n" for i, persona in enumerate(implicit_persona_list)]
            implicit_persona_str = "".join(implicit_persona_list)

        # explicit prompt
        if explicit_persona:
            explicit_persona_list = []
            for persona in explicit_persona:
                for key, value in persona.items():
                    explicit_persona_list.append(f"{key}: {value}")
            explicit_persona_str = "\n".join(explicit_persona_list)


        # question prompt
        choices = [f"{OUTPUT_MAP[i]}.{choice}\n" for i, choice in enumerate(choices)]
        choice = "".join(choices)

        # implicit + explicit
        if explicit_persona and implicit_persona:
            prompt = \
f"""A person can be described as follows:

{explicit_persona_str}

The person has the following opinions on {topic}.

Opinions:
{implicit_persona_str}
Based on the above list of opinions and the demographic information, which answer choice will this person select for the question:

Question: {question}

Answer choices:
{choice}
Answer:
"""
        # implicit prompt
        elif implicit_persona:
            prompt = \
f"""A person has the following opinions on {topic}.

Opinions:
{implicit_persona_str}
Based on the above list of opinions, which answer choice will this person select for the question:

Question: {question}

Answer choices:
{choice}
Answer:
"""
            # explicit prompt
        elif explicit_persona:
            prompt = \
f"""A person can be described as follows:

{explicit_persona_str}

Based on the demographic information, which answer choice will this person select for the question:

Question: {question}

Answer choices:
{choice}
Answer:
"""

        return prompt

    def __call__(self, implicit_persona: List[str], explicit_persona: List[str],
                 topic:str, question: str, choices: List[str]) -> str:
        generation_query = self.make_query(
            implicit_persona=implicit_persona,
            explicit_persona=explicit_persona,
            topic=topic,
            question=question,
            choices=choices,
        )

        generated_sent = self.openai_wrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=1,
            stop_token="###",
            temperature=0.0
        )
        return generated_sent.strip()  # gpt3 turbo adds newline in the beginning so strip it.


class PersonalizedQA:
    """
    save personalized opinion to output json.
    """
    def __init__(self, args):
        self.num_implicit = args.num_implicit

    def personalized_qa(self, persona: PersonaCreator, user_responses_jsons: Dict, option: int, max_users:int=None):
        """ get answer choice based on implicit/explicit/implicit+explicit persona.
        """
        model_generated = []
        curr_user_num = 0
        for user_response_json in tqdm(user_responses_jsons, desc="processing user response #"):
            curr_user_num += 1
            if max_users and curr_user_num >= max_users:
                break
            user_id = user_response_json["user_id"]
            topic = user_response_json["topic"]

            if self.num_implicit > len(user_response_json["implicit_persona"]):
                num_implicit = len(user_response_json["implicit_persona"])
            else:
                num_implicit = self.num_implicit

            implicit_persona = user_response_json["implicit_persona"][:num_implicit] if option == 0 or option == 2 else None
            explicit_persona = user_response_json["explicit_persona"] if option == 1 or option == 2 else None

            generated_output = []
            for persona_qa in user_response_json["implicit_questions"]:
                model_choice = persona(
                    implicit_persona=implicit_persona,
                    explicit_persona=explicit_persona,
                    topic=topic,
                    question=persona_qa["question"],
                    choices=persona_qa["choices"],
                )
                user_choice = persona_qa["answer"]
                choice_idx = persona_qa["choices"].index(user_choice)
                user_choice = OUTPUT_MAP[choice_idx]
                generated_output.append({
                    "model_choice": model_choice,
                    "user_choice": user_choice,
                    "qid": persona_qa["qid"],
                })

            model_generated.append({"user_id": user_id, "topic": topic, "generated_output": generated_output})
        return model_generated

def calculate_accuracy(model_generation_path):
    model_generation = read_jsonl_or_json(model_generation_path)
    print("================ model_generation_path: {} ================".format(model_generation_path))
    accuracy_list = []
    user_accuracy_list = []
    for model_output in model_generation:
        user_id = model_output["user_id"]
        generated_output = model_output["generated_output"]

        correct, incorrect = 0, 0
        for response in generated_output:
            model_choice = response['model_choice']
            user_choice = response['user_choice']

            if user_choice == model_choice:
                correct += 1
            else:
                incorrect += 1
        accuracy_per_user = correct / len(generated_output)
        user_accuracy_list.append({user_id: accuracy_per_user})
        accuracy_list.append(accuracy_per_user)
    final_accuracy = sum(accuracy_list) / len(accuracy_list)
    return {"accuracy": final_accuracy, "user-accuracy": user_accuracy_list}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="data/opinionqa/sampled_user_responses_decl.json", help="json path")
    parser.add_argument("--out_dir", type=str, default="data/model-output/", help="json path")
    parser.add_argument("--cache_path", type=str, default="data/cache/gpt_cache.jsonl", help="json path")
    parser.add_argument("--num_implicit", type=int, default=2, help="number of implicit persona to use")
    parser.add_argument("--max_users_for_eval", type=int, default=100, help="max num users to do inference on. Each user has about 20 questions.")
    parser.add_argument("--option", type=int, default=0, choices=[0, 1, 2], help="0: implicit, 1: explicit, 2: both")
    args = parser.parse_args()

    persona = PersonaCreator(engine="text-davinci-003", openai_wrapper=OpenAIWrapper(cache_path=args.cache_path))
    output = PersonalizedQA(args).personalized_qa(
        persona=persona, user_responses_jsons=read_jsonl_or_json(args.in_path), option=args.option, max_users=args.max_users_for_eval
    )

    if args.option == 0:
        dir_name = "implicit"
    if args.option == 1:
        dir_name = "explicit"
    if args.option == 2:
        dir_name = "imp_exp"

    output_dir = os.path.join(args.out_dir, dir_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, "model_generation.json")
    write_json(outpath=output_file, json_data=output)

    metrics = calculate_accuracy(output_file)
    metrics_file = os.path.join(output_dir, "model_accuracy.json")
    write_json(outpath=metrics_file, json_data=metrics)
