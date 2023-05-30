#python personalized_opinionqa/personalized_opinionqa.py --option 0 --num_implicit 2 --max_users 100
#python personalized_opinionqa/personalized_opinionqa.py --option 1
#python personalized_opinionqa/personalized_opinionqa.py --option 2 --num_implicit 2 --max_users 100

#python personalized_opinionqa/personalized_opinionqa.py --option 0  --num_implicit 8 --max_users 20
#python personalized_opinionqa/personalized_opinionqa.py --option 2  --num_implicit 8 --max_users 20

#python personalized_opinionqa/personalized_opinionqa.py --option 0  --num_implicit 8 --max_users 100
#python personalized_opinionqa/personalized_opinionqa.py --option 2  --num_implicit 8 --max_users 100

#python personalized_opinionqa/personalized_opinionqa.py --option 0  --num_implicit 16 --max_users 100
#python personalized_opinionqa/personalized_opinionqa.py --option 2  --num_implicit 16 --max_users 100

#python personalized_opinionqa/personalized_opinionqa.py --option -1  --num_implicit 16 --max_users 100



#python personalized_opinionqa/personalized_opinionqa.py --option 2  --num_implicit 8 --max_users 35 --max_ques 30 --max_topics -1
#python personalized_opinionqa/personalized_opinionqa.py --option 1  --num_implicit 8 --max_users 35 --max_ques 30 --max_topics -1
#python personalized_opinionqa/personalized_opinionqa.py --option 0  --num_implicit 8 --max_users 35 --max_ques 30 --max_topics -1
#python personalized_opinionqa/personalized_opinionqa.py --option -1  --num_implicit 8 --max_users 35 --max_ques 30 --max_topics -1

#python personalized_opinionqa/personalized_opinionqa.py --option 2  --num_implicit 16 --max_users 35 --max_ques 30 --max_topics -1
#python personalized_opinionqa/personalized_opinionqa.py --option 0  --num_implicit 16 --max_users 35 --max_ques 30 --max_topics -1

#python personalized_opinionqa/demography_prediction.py --predict "ideo" --num_implicit 16 --max_users 35 --max_topics -1
#python personalized_opinionqa/demography_prediction.py --predict "ideo" --num_implicit 8 --max_users 35 --max_topics -1


#python personalized_opinionqa/mean_responses_qa.py --experiment_type "ideology:Republican"
#python personalized_opinionqa/mean_responses_qa.py --experiment_type "ideology:Democrat"
#python personalized_opinionqa/mean_responses_qa.py --experiment_type "ideology:Independent"
#python personalized_opinionqa/mean_responses_qa.py --experiment_type "overall"


#python personalized_opinionqa/personalized_opinionqa.py --in_path "data/opinionqa/sampled_user_responses_20_decl_topk_compressed.jsonl" --out_dir "data/model-output-topk/" --option 2 --implicit_sampling "topk" --num_implicit 8 --max_users 35 --max_ques 30 --max_topics -1 --cache_path "data/cache/gpt_cache_topk.jsonl"

# topk implicit neighbors related experiments.
#python personalized_opinionqa/personalized_opinionqa.py --explicit_definition "ideo" --in_path "data/opinionqa/sampled_user_responses_20_decl_topk_compressed.jsonl" --out_dir "data/model-output-topk/" --option 2 --implicit_sampling "topk" --num_implicit 8 --max_users 35 --max_ques 30 --max_topics -1 --cache_path "data/cache/gpt_cache_topk.jsonl"
#python personalized_opinionqa/personalized_opinionqa.py --explicit_definition "demo" --in_path "data/opinionqa/sampled_user_responses_20_decl_topk_compressed.jsonl" --out_dir "data/model-output-topk/" --option 2 --implicit_sampling "topk" --num_implicit 8 --max_users 35 --max_ques 30 --max_topics -1 --cache_path "data/cache/gpt_cache_topk.jsonl"
#python personalized_opinionqa/personalized_opinionqa.py --in_path "data/opinionqa/sampled_user_responses_20_decl_topk_compressed.jsonl" --out_dir "data/model-output-topk/" --option 2 --implicit_sampling "topk" --num_implicit 3 --max_users 35 --max_ques 30 --max_topics -1 --cache_path "data/cache/gpt_cache_topk.jsonl"
#python personalized_opinionqa/personalized_opinionqa.py --in_path "data/opinionqa/sampled_user_responses_20_decl_topk_compressed.jsonl" --out_dir "data/model-output-topk/" --option 0 --implicit_sampling "topk" --num_implicit 3 --max_users 35 --max_ques 30 --max_topics -1 --cache_path "data/cache/gpt_cache_topk.jsonl"
python personalized_opinionqa/personalized_opinionqa.py --in_path "data/opinionqa/sampled_user_responses_20_decl_topk_compressed.jsonl" --out_dir "data/model-output-topk/" --option 0 --implicit_sampling "topk" --num_implicit 8 --max_users 35 --max_ques 30 --max_topics -1 --cache_path "data/cache/gpt_cache_topk.jsonl"