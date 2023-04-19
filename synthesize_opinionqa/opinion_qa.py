import argparse
import json
from typing import List

from gptinference.base_prompt import Prompt
from gptinference.openai_wrapper import OpenAIWrapper


class AbstractTakeawayForClaimTask(Prompt):
    def __init__(self, engine: str, openai_wrapper: OpenAIWrapper):
        super().__init__()
        self.openai_wrapper = openai_wrapper
        self.engine = engine
        self.output_mapping = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def make_query(self, implicit_persona: List[str], explicit_persona: List[str], question: List[str]) -> str:
        if not implicit_persona or not explicit_persona or not question:
            return ""

        # implicit prompt
        if explicit_persona is None:
            implicit_persona_list = []
            for persona in implicit_persona:
                implicit_persona_list.append(persona["declarative"]) # "declarative"
            implicit_persona_str = "\n".join(implicit_persona_list)

        # explicit prompt
        if implicit_persona is None:
            explicit_persona_list = []
            for persona in explicit_persona:
                for key, value in persona:
                    explicit_persona_list.append(f"{key}: {value}")
            explicit_persona_str = "\n".join(explicit_persona_list)


        # question prompt
        test_prompts = []
        test_answers = []
        test_choices = []
        for q_item in question:
            question = q_item["question"].strip()
            choices = q_item["choices"]
            answer = q_item["answer"]

            test_choices.append(choices)
            choices = [f"{self.output_mapping[i]}.{choice}\n" for i, choice in enumerate(choices)]
            choice = "".join(choices)

            test_prompt = f"Question: {question}\n{choice}Answer:"

            # implicit prompt
            if explicit_persona is None:
                prompt = implicit_persona_str + "\nHow would this person answer the following question?\n" + test_prompt

            # explicit prompt
            if implicit_persona is None:
                prompt = "This person can be described as follows:\n" + explicit_persona_str \
                         + "\nHow would this person answer the following question?\n" + test_prompt
            test_prompts.append(prompt)
            test_answers.append(answer)

        return test_prompts, test_answers, test_choices
#         question_prefix_template = \
#             f"""
# Claim: {claim}
#
# Abstract: {abstract}
#
# Now, answer these two questions:
# Q1. Is the claim and abstract related or unrelated?
# Q2. How can someone accurately extract the main point of the abstract in relation to the claim?(Only extract detail about the salient relation. Do NOT provide any stance about the claim. )
# """
#         query = f"""{self.question_prefix}{question_prefix_template.format(claim=claim, abstract=abstract)}"""
#         query = f"{query}{self.intra_example_sep}"
#         return query

    def __call__(self, implicit_persona: List[str], explicit_persona: List[str], question: List[str]) -> str:
        test_prompts, test_answers, test_choices = self.make_query(
            implicit_persona=implicit_persona, explicit_persona=explicit_persona, question=question
        )

        generated_sent = self.openai_wrapper.call_batch(
            prompt=test_prompts,
            engine=self.engine,
            max_tokens=1,
            stop_token="###",
            temperature=0.0,
        )

        # # (extract answers) A1.xxx\n\nA2.xxx
        # generated_sent = generated_sent.strip()  # gpt3 turbo adds newline in the beginning so strip it.
        # generated_answers = generated_sent.split("\n\n")
        # if len(generated_answers) != 2:
        #     # second attempt
        #     generated_answers = generated_sent.split("\n")
        # 
        # # first relevant_sent is just "A2. " so ignore it.
        # relation = ""
        # takeaway_sent = ""
        # try:
        #     relation=generated_answers[0].strip()
        #     takeaway_sent=generated_answers[1].strip()
        #     # Make the abstract takeaways txt cleaner. (remove: Q2. The revised claim could be: )
        #     # {'A0': 'Q2. The revised claim could be: "Delayed diagnosis of cervical cancer is a major contributor to increasing rates of cervical cancer in Ethiopia."', 'A1': 'Q2. The claim can be rewritten to: Cervical cancer rates have increased in Ethiopia since the launch of the Gynecologic Oncology Fellowship Training Program at St. Paul’s Hospital Millennium Medical college in 2016.', 'A2': 'Q2. The claim can be rewritten to: "Cervical cancer screening practice among age-eligible women in Wolaita Zone hospitals in Southern Ethiopia is low, despite age, being an adherence supporter, source of information from health care professionals, history of multiple sexual partners, sexually transmitted infection, knowledge and attitude being important predictors of cervical cancer screening practice."', 'A3': 'Q2. The revised claim could be: "Cervical cancer screening and treatment services in South West Shoa Zone of Oromia Region, Ethiopia, have revealed an increasing rate of cervical cancer cases."', 'A4': 'Q2. The claim can be rewritten to: "Cervical cancer screening practices and associated factors among females of reproductive age in Durame, Southern Ethiopia are increasing."', 'A5': 'Q2. The rewritten claim could be: "The utilization of cervical cancer screening services and its predictors among eligible women in Ethiopia are being assessed in a systematic review and meta-analysis."'}
        #     takeaway_sent = " ".join(takeaway_sent.split(":" if ":" in takeaway_sent else ".")[1:])
        # except Exception as exc:
        #     print(f"Exception caught in extracting rel or sents in claim abstract link: {exc}.\n"
        #           f"Could not extract from generated text: {generated_sent}")
        # 
        # return relation, takeaway_sent


def load_dataset(json_path):
    with open(json_path, "r") as fd:
        print(f"loading {json_path}...")
        user_responses = json.load(fd)
    return user_responses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="data/opinionqa/sampled_user_responses.json", help="json path")
    parser.add_argument("--num_implicit", type=int, default=1, help="number of implicit persona to use")
    args = parser.parse_args()


    openai_wrapper = OpenAIWrapper(cache_path="cache.jsonl")
    gpt = AbstractTakeawayForClaimTask(engine="text-davinci-003", openai_wrapper=openai_wrapper)

    user_responses = load_dataset(args.json_path)

    for user in user_responses:
        user_id = user['user_id']
        survey = user['survey']
        topic = user['topic']
        explicit_persona = user['explicit_persona']
        implicit_persona = user['implicit_persona'][:args.num_implicit]
        implicit_questions = user['implicit_questions']

        gpt(implicit_persona=implicit_persona, explicit_persona=None, question=implicit_questions)
        # gpt(implicit_persona=None, explicit_persona=explicit_persona, question=implicit_questions)
        # gpt(implicit_persona=implicit_persona, explicit_persona=explicit_persona, question=implicit_questions)


    # sample_implicit_persona = "I never used air guns such as paintball, BB or pellet guns when I was growing up."
    # sample_question = "Have you ever been a victim of a violent crime, whether a gun was used or not?"
    # sample_answer_choices = "A.Yes\nB.No\nC.Refused"
    # print(f"implicit persona: {sample_implicit_persona}\nquestion: {sample_question}\nanswer choices: {sample_answer_choices}")
    # print(gpt(implicit_persona=sample_implicit_persona, question=sample_question, answer_choices=sample_answer_choices))
