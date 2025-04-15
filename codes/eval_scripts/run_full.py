import sys
import subprocess

date = sys.argv[1]
benchmark = sys.argv[2]

subprocess.run(["python", "../datasets/make_datasets.py", benchmark])
subprocess.run(["python", "./run_baseline_wo_retrieve.py", "ChatGPT_request", date, "full", benchmark])
subprocess.run(["python", "./run_baseline_rag.py", "ChatGPT_request", date, "full", "[20]", "[0,20,40,60,80,100]", benchmark])
subprocess.run(["python", "./run_baseline_compress.py", "ChatGPT_request", date, "full", "[20]", "[0,20,40,60,80,100]", benchmark])
subprocess.run(["python", "./run_baseline_compress.py", date, "full", "ChatGPT_request", "[20]", "[0,20,40,60,80,100]", "1110", "dynamic", benchmark])