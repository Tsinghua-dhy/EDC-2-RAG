import sys
import subprocess
eval_model = sys.argv[1]#llama3_request, GPT_Instruct_request, ChatGPT_request
date = sys.argv[2]
dataset = sys.argv[3]#500 or 113
benchmark = sys.argv[4]
#python run_baseline_wo_retriev.py ChatGPT_request 0415 full hotpotqa

print("start_to_run")
print("start_to_eval")
subprocess.run(["python", "../run_methods/eval_baseline_wo_retrieve.py", eval_model, date, dataset,benchmark])
print("end_eval")
print("start_to_extracte_answer")
if eval_model == "llama3_request":
    eval_method = "eval_llama3"
elif eval_model == "GPT_Instruct_request":
    eval_method = "eval_3.5instruct"
elif eval_model == "GPT4o_request":
    eval_method = "eval_4o"
else:
    eval_method = "eval_3.5turbo"
subprocess.run(["python", "../eval_metric/extracted_answer_single.py", date, dataset, eval_method,benchmark])
print("end_extracte_answer")
print("start_to_caculate_F1_EM")
subprocess.run(["python", "../eval_metric/caculate_F1_EM_single.py", date, dataset, eval_method,benchmark])
print("end_caculate_F1_EM")