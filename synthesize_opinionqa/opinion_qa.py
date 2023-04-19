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

class PersonalizedOpinionGPT(Prompt):
    def __init__(self, engine: str, openai_wrapper: OpenAIWrapper):
        super().__init__()
        self.openai_wrapper = openai_wrapper
        self.engine = engine

    def make_query(self, implicit_persona: List[str], explicit_persona: List[str],
                   question: List[str], choices: List[str]) -> str:
        if not question or not choices:
            return ""

        # implicit prompt
        if implicit_persona:
            implicit_persona_list = []
            for persona in implicit_persona:
                implicit_persona_list.append(persona["declarative_opinion"])
            implicit_persona_str = " and ".join(implicit_persona_list)

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
f"""This person can be described as follows:

{explicit_persona_str}

This person thinks that {implicit_persona_str}

How would this person answer the following question?

Question: {question}

{choice}
Answer:
"""
        # implicit prompt
        elif implicit_persona:
            prompt = \
f"""This person thinks that {implicit_persona_str}

How would this person answer the following question?

Question: {question}

{choice}
Answer:
"""
            # explicit prompt
        elif explicit_persona:
            prompt = \
f"""This person can be described as follows:

{explicit_persona_str}

How would this person answer the following question?

Question: {question}

{choice}
Answer:
"""

        return prompt

    def __call__(self, implicit_persona: List[str], explicit_persona: List[str], question: str, choices: List[str]) -> str:
        generation_query = self.make_query(
            implicit_persona=implicit_persona,
            explicit_persona=explicit_persona,
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


class PersonalizedOpinionSaver():
    """
    save personalized opinion to output json.
    """
    def __init__(self, args):
        self.num_implicit = args.num_implicit

    def add_implicit_persona(self, persona: PersonalizedOpinionGPT, user_responses_jsons: Dict):
        """ get answer choice based on implicit persona.
        "implicit_persona": [
        {
           "qid": "GUNKILLF2_W26",
           "question": "Thinking about people who commit suicide using a gun, which comes closer to your view, even if neither is exactly right?",
           "choices": [
               "They would find a way to do it whether they had access to a gun or not",
               "They would be less likely to do it if they didn't have access to a gun",
               "Refused"
               ],
           "answer": "They would be less likely to do it if they didn't have access to a gun",
           "declarative_opinion": "xxxx",
           "subtopic_cg": [
               "crime/security"
               ]
        },
        ...
        ]
        """
        model_generated = []
        for user_response_json in tqdm(user_responses_jsons[:2], desc="processing user response #"):
            user_id = user_response_json["user_id"]
            implicit_persona = user_response_json["implicit_persona"][:self.num_implicit]
            generated_output = []
            for persona_qa in user_response_json["implicit_questions"][:2]:
                model_choice = persona(
                    implicit_persona=implicit_persona,
                    explicit_persona=None,
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

            model_generated.append({"user_id": user_id, "generated_output": generated_output})
        return model_generated


    def add_explicit_persona(self, persona: PersonalizedOpinionGPT, user_responses_jsons: Dict):
        """ get answer choice based on explicit persona.
        "explicit_persona": [
            {"Age": "50-64"},
            {"Citizenship": "Yes"},
            {"Region": "West"},
            {"Education": "Postgraduate"},
            {"Income": "Less than $30,000"},
            {"Marital status": "Living with a partner"},
            {"Political ideology": "Liberal"},
            {"Political party": "Democrat"},
            {"Race": "White"},
            {"Religion": "Roman Catholic"},
            {"Frequency of religious attendance": "Seldom"},
            {"Gender": "Female"}
        ],
        """
        model_generated = []
        for user_response_json in tqdm(user_responses_jsons[:2], desc="processing user response #"):
            user_id = user_response_json["user_id"]
            explicit_persona = user_response_json["explicit_persona"]
            generated_output = []
            for persona_qa in user_response_json["implicit_questions"][:2]:
                model_choice = persona(
                    implicit_persona=None,
                    explicit_persona=explicit_persona,
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

            model_generated.append({"user_id": user_id, "generated_output": generated_output})
        return model_generated

    def add_both_persona(self, persona: PersonalizedOpinionGPT, user_responses_jsons: Dict):
        """
        get answer choice based on implicit and explicit persona.
        """
        model_generated = []
        for user_response_json in tqdm(user_responses_jsons[:2], desc="processing user response #"):
            user_id = user_response_json["user_id"]
            implicit_persona = user_response_json["implicit_persona"][:self.num_implicit]
            explicit_persona = user_response_json["explicit_persona"]
            generated_output = []
            for persona_qa in user_response_json["implicit_questions"][:2]:
                model_choice = persona(
                    implicit_persona=implicit_persona,
                    explicit_persona=explicit_persona,
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

            model_generated.append({"user_id": user_id, "generated_output": generated_output})
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
    print("accuracy per user: {}".format(user_accuracy_list))
    print("final average accuracy: {}\n".format(final_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="../data/opinionqa/sampled_user_responses_decl.json", help="json path")
    parser.add_argument("--out_dir", type=str, default="../data/opinionqa/model-output/", help="json path")
    parser.add_argument("--cache_path", type=str, default="../data/cache/gpt-davinci_cache.jsonl", help="json path")
    parser.add_argument("--num_implicit", type=int, default=1, help="number of implicit persona to use")
    args = parser.parse_args()

    persona = PersonalizedOpinionGPT(engine="text-davinci-003", openai_wrapper=OpenAIWrapper(cache_path=args.cache_path))
    implicit_output = PersonalizedOpinionSaver(args).add_implicit_persona(persona=persona, user_responses_jsons=read_jsonl_or_json(args.in_path))
    explicit_output = PersonalizedOpinionSaver(args).add_explicit_persona(persona=persona, user_responses_jsons=read_jsonl_or_json(args.in_path))
    imp_exp_output = PersonalizedOpinionSaver(args).add_both_persona(persona=persona, user_responses_jsons=read_jsonl_or_json(args.in_path))

    implicit_output_dir = os.path.join(args.out_dir, "implicit")
    explicit_output_dir = os.path.join(args.out_dir, "explicit")
    imp_exp_output_dir = os.path.join(args.out_dir, "imp_exp")

    Path(implicit_output_dir).mkdir(parents=True, exist_ok=True)
    Path(explicit_output_dir).mkdir(parents=True, exist_ok=True)
    Path(imp_exp_output_dir).mkdir(parents=True, exist_ok=True)

    implicit_output_file = os.path.join(implicit_output_dir, "model_generation.json")
    explicit_output_file = os.path.join(explicit_output_dir, "model_generation.json")
    imp_exp_output_file = os.path.join(imp_exp_output_dir, "model_generation.json")

    write_json(outpath=implicit_output_file, json_data=implicit_output)
    write_json(outpath=explicit_output_file, json_data=explicit_output)
    write_json(outpath=imp_exp_output_file, json_data=imp_exp_output)

    calculate_accuracy(implicit_output_file)
    calculate_accuracy(explicit_output_file)
    calculate_accuracy(imp_exp_output_file)