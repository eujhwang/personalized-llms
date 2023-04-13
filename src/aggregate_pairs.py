import ast
import itertools


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
            if len(a) <= 1:
                continue
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
    print("total_responses:", total_responses)
    print("total_pairs:", len(pairs))


    return demo_to_question_dict
