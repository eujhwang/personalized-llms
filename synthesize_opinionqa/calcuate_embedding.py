from typing import Dict

import faiss
import numpy as np
import openai
import pandas as pd
from gptinference.base_prompt import Prompt
from gptinference.openai_wrapper import OpenAIWrapper
from gptinference.utils import write_json, read_jsonl_or_json
from tqdm import tqdm


class EmbeddingGenerator(Prompt):
    """
    Creates "declarative opinion" using "question", "answer", "choices".

    Question: How often, if ever, did you use air guns, such as paintball, BB or pellet guns when you were growing up?
    Answer: Never.
    Declarative opinion: I never used air guns such as paintball, BB or pellet guns when I was growing up.
    """
    def __init__(self, engine: str, encoding: str, max_tokens: int, openai_wrapper: OpenAIWrapper):
        super().__init__()
        self.openai_wrapper = openai_wrapper
        self.engine = engine
        self.encoding = encoding
        self.max_tokens = max_tokens

    def get_embedding(self, text):
        text = text.strip()
        return openai.Embedding.create(input=[text], model=self.engine)['data'][0]['embedding']

    def __call__(self, text: str) -> str:
        emb = self.get_embedding(text) # dim size: 1536
        return emb
        # generation_query = self.make_query(question=question, answer=answer)
        # generated_decl_sent = self.openai_wrapper.call(
        #     prompt=generation_query,
        #     engine=self.engine,
        #     max_tokens=30,
        #     stop_token="###",
        #     temperature=0.0
        # )
        # return generated_decl_sent.strip()  # gpt3 turbo adds newline in the beginning so strip it.

class EmbeddingSaver:
    """
    Adds "declarative_opinion" to implicit persona json.
    """

    def generate_embedding(self, generator: EmbeddingGenerator, user_responses_jsons: Dict):
        """ For every implicit persona of every user, add embedding of declarative opinion
        """
        for user_response_json in tqdm(user_responses_jsons[:1], desc="processing user response #"):
            for persona_qa in user_response_json["implicit_persona"][:5]:
                emb = generator(text=persona_qa["declarative_opinion"])
                persona_qa["emb"] = emb

            for persona_qa in user_response_json["implicit_questions"][:2]:
                emb = generator(text=persona_qa["question"])
                persona_qa["emb"] = emb

        return user_responses_jsons

    def faiss_similarity_score(self, generator: EmbeddingGenerator, user_responses_jsons: Dict, topk: int):
        # user_responses_jsons = self.generate_embedding(generator=generator, user_responses_jsons=user_responses_jsons)

        for user_response_json in tqdm(user_responses_jsons[:1], desc="processing user response #"):
            decl_op, decl_emb = [], []
            for persona_qa in user_response_json["implicit_persona"][:5]:
                decl_op.append(persona_qa["declarative_opinion"])
                decl_emb.append(persona_qa["emb"])
            decl_emb = np.array(decl_emb, dtype='float32') # (5, 1536)

            # faiss indexing for all declarative opinions
            dim = decl_emb.shape[1]
            index = faiss.IndexFlatL2(dim)
            faiss.normalize_L2(decl_emb)
            index.add(decl_emb)

            topk_op = None
            # for each implicit question, find the nearest opinions using faiss search index
            for persona_qa in user_response_json["implicit_questions"][:2]:
                _question_emb = persona_qa["emb"]
                question_emb = np.array([_question_emb], dtype='float32')
                faiss.normalize_L2(question_emb)

                # search for all nearest neighbours
                _distances, _ann = index.search(question_emb, k=topk)
                distances = _distances[0].tolist()
                ann = _ann[0].tolist()

                # get topk declarative opinions
                topk_op = [[decl_op[anno_index], distances[i]] for i, anno_index in enumerate(ann)]
            
            persona_qa["topk_opinions"] = topk_op

        return user_responses_jsons


if __name__ == '__main__':
    in_path = "../data/opinionqa/sampled_user_responses_decl.json"
    out_path1 = "../data/opinionqa/sampled_user_responses_decl_emb.json"
    out_path2 = "../data/opinionqa/sampled_user_responses_decl_topk.json"
    cache_path="../data/cache/emb_cache.jsonl"
    generator = EmbeddingGenerator(engine="text-embedding-ada-002", encoding="cl100k_base", max_tokens=512, openai_wrapper=OpenAIWrapper(cache_path=cache_path))
    enhanced_json_with_embedding = EmbeddingSaver().generate_embedding(
        generator=generator, user_responses_jsons=read_jsonl_or_json(in_path))
    write_json(outpath=out_path1, json_data=enhanced_json_with_embedding)

    enhanced_json_with_topk = EmbeddingSaver().faiss_similarity_score(
        generator=generator, user_responses_jsons=enhanced_json_with_embedding, topk=5)
    write_json(outpath=out_path2, json_data=enhanced_json_with_topk)