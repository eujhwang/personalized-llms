import json

import torch.cuda
import argparse
import numpy as np
import pandas as pd
import os


PEW_SURVEY_LIST = [26, 27, 29, 32, 34, 36, 41, 42, 43, 45, 49, 50, 54, 82, 92]

MODEL_NAMES = {
    # 'human max': 'human (worst)',
    # 'human mean': 'human (avg)',
    # 'random': 'random',
    # 'ai21_j1-grande': 'j1-grande',
    # 'ai21_j1-jumbo': 'j1-jumbo',
    # 'ai21_j1-grande-v2-beta': 'j1-grande-v2-beta',
    # 'openai_ada': 'ada',
    # 'openai_davinci': 'davinci',
    # 'openai_text-ada-001': 'text-ada-001',
    # 'openai_text-davinci-001': 'text-davinci-001',
    # 'openai_text-davinci-002': 'text-davinci-002',
    'openai_text-davinci-003': 'text-davinci-003',
}

MODEL_ORDER = {k: ki for ki, k in enumerate(MODEL_NAMES.keys())}

pd.options.display.max_columns = 15
# pd.options.display.max_rows = 999

def get_probabilities(lps, references, mapping):
    min_prob = np.exp(np.min(list(lps.values())))
    remaining_prob = max(0, 1 - sum([np.exp(v) for v in lps.values()]))

    dist, misses = [], []
    for ref in references:
        prefix = mapping[ref]
        values = [lps[key] for key in [f" {prefix}", prefix] if key in lps]
        misses.append(len(values) == 0)
        dist.append(np.max(values) if len(values) else None)

    Nmisses = sum(misses)
    if Nmisses > 0:
        miss_value = np.log(min(min_prob, remaining_prob / Nmisses))
        dist = [d if d is not None else miss_value for d in dist]

    probs_unnorm = np.array([np.exp(v) for v in dist])

    res = {'logprobs': dist,
           'probs_unnorm': probs_unnorm,
           'probs_norm': probs_unnorm / np.sum(probs_unnorm),
           'misses': misses}

    return res


def extract_model_opinions(result_instance, context_type, info_df):
    row = {}

    input_id = result_instance['instance']['id']
    question_raw = result_instance['instance']['input']['text']
    references = [r['output']['text'] for r in result_instance['instance']['references']]
    mapping = result_instance['output_mapping']
    if context_type not in ['steer-portray', 'steer-bio']:
        context = result_instance['request']['prompt'].split(f"Question: {question_raw}")[0].strip()
    else:
        context = question_raw.split('Question:')[0].strip() + '\n'
        question_raw = question_raw.replace(context, "").strip().replace('Question: ', '')
    question = question_raw + f" [{'/'.join(references)}]"

    top_k_logprobs = result_instance['result']['completions'][0]['tokens'][0]['top_logprobs']

    for k, v in zip(['input_id', 'question_raw', 'question', 'references',
                     'context', 'mapping', 'top_k_logprobs'],
                    [input_id, question_raw, question, references, context, mapping, top_k_logprobs]):
        row[k] = v

    ## Get probability distribution

    info_loc = np.where(np.logical_and(info_df['question'] == question_raw,
                                       [set(r) == set(references) for r in info_df['references']]))[0]
    assert len(info_loc) == 1

    info = info_df.iloc[info_loc]
    ordinal = info['option_ordinal'].values[0]
    ordinal_refs = info['references'].values[0][:len(ordinal)]
    refusal_refs = info['references'].values[0][len(ordinal):]

    dist_info = get_probabilities(top_k_logprobs, info['references'].values[0], {v: k for k, v in mapping.items()})
    dist_info['D_M'] = dist_info['probs_unnorm'][:len(ordinal)] / np.sum(dist_info['probs_unnorm'][:len(ordinal)])
    dist_info['R_M'] = np.sum(dist_info['probs_norm'][len(ordinal):])
    dist_info['ordinal'] = ordinal
    dist_info['ordinal_refs'] = ordinal_refs
    dist_info['refusal_refs'] = refusal_refs
    dist_info['qkey'] = info['key'].values[0]

    row.update(dist_info)

    return row

def get_model_opinions(result_dir, result_files, info_df):
    model_df = []
    for f in result_files:
        context_type = f.split('context=')[1].split(',')[0]
        model_name = f.split('model=')[1].split(',')[0]
        print(f)
        print(model_name, context_type)

        results_json = json.load(open(os.path.join(result_dir, f, 'scenario_state.json'), 'rb'))['request_states']
        mdf = pd.DataFrame([extract_model_opinions(r, context_type, info_df) for r in results_json])

        mdf['results_path'] = f
        mdf['context_type'] = context_type
        mdf['model_name'] = MODEL_NAMES[model_name]
        mdf['model_order'] = MODEL_ORDER[model_name]
        model_df.append(mdf)

        print('-' * 100)
    model_df = pd.concat(model_df)
    return model_df


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load topic-mapping.npy
    topic_mapping = np.load('data/topic_mapping.npy', allow_pickle=True)
    topic_mapping = topic_mapping.tolist()
    print(len(topic_mapping), type(topic_mapping), )
    # print(topic_mapping)

    DATASET_DIR = os.path.join(args.data_dir, 'human_resp')
    RESULT_DIR = os.path.join(args.data_dir, 'runs')
    CONTEXT = "default"
    if CONTEXT == "default":
        SURVEY_LIST = [f'American_Trends_Panel_W{SURVEY_WAVE}' for SURVEY_WAVE in PEW_SURVEY_LIST] + \
                      ['Pew_American_Trends_Panel_disagreement_500']

    SURVEY_LIST = SURVEY_LIST[:1]
    print("SURVEY_LIST:", len(SURVEY_LIST), SURVEY_LIST)

    for SURVEY_NAME in SURVEY_LIST:
        RESULT_FILES = [f for f in os.listdir(RESULT_DIR) if SURVEY_NAME in f and f'context={CONTEXT}' in f and 'openai_text-davinci-003' in f]
        print("RESULT_FILES:", RESULT_FILES)

        ## Read human responses and survey info
        info_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'info.csv'))
        info_df['option_ordinal'] = info_df.apply(lambda x: eval(x['option_ordinal']), axis=1)
        info_df['references'] = info_df.apply(lambda x: eval(x['references']), axis=1)

        md_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'metadata.csv'))
        md_df['options'] = md_df.apply(lambda x: eval(x['options']), axis=1)
        md_order = {'Overall': {'Overall': 0}}
        md_order.update({k: {o: oi for oi, o in enumerate(opts)} for k, opts in zip(md_df['key'], md_df['options'])})

        print("info_df:", info_df)
        print("md_df:", md_df)
        print("md_order:", md_order)

        ## Get human opinion distribution
        if SURVEY_NAME != "Pew_American_Trends_Panel_disagreement_500":
            resp_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'responses.csv'))
            human_df = pd.concat([extract_human_opinions(resp_df,
                                                            model_df,
                                                            md_df,
                                                            demographic=demographic,
                                                            wave=int(SURVEY_NAME.split('_W')[1]))
                                  for demographic in ph.DEMOGRAPHIC_ATTRIBUTES])

# for key, value in topic_mapping.items():
    #     print("key:", key)
    #     print("value:", value)
    #     break

    # load human_resp
    # resp_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'responses.csv'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data/opinion-qa/', help="start collecting memes from reddit")
    args = parser.parse_args()

    main(args)
