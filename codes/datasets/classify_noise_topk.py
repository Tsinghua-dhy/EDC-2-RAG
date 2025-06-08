import json
import copy
import sys

dataset = sys.argv[1]

input_file = f"your_path/{dataset}/datasets/{dataset}_results_w_negative_passages_full_embedding.json"

with open(input_file, "r", encoding="utf-8") as lines:
    cases = json.load(lines)
    for topk in [5,10,20,30,50,70,100]:
        for noise in [0, 20, 40, 60, 80, 100]:
            outs = []
            output_file = f"your_path/{dataset}/datasets/{dataset}_results_random_full_w_negative_passages_noise{noise}_topk{topk}_embedding.json"
            for case in cases:
                out = copy.deepcopy(case)
                out["passages"] = []
                n = topk*noise//100
                p = topk - n
                ii = 0
                for passage in out["positive_passages"]:
                    if ii < p:
                        out["passages"].append(passage)
                        ii += 1
                    else:
                        break
                ii = 0
                for passage in out["negative_passages"]:
                    if ii < n:
                        out["passages"].append(passage)
                        ii += 1
                    else:
                        break
                if "negative_passages" in out:
                    del out["negative_passages"]
                if "positive_passages" in out:
                    del out["positive_passages"]
                out["passages"] = sorted(out["passages"], key=lambda x: x["score"], reverse=True)
                outs.append(out)
            with open(output_file, "w", encoding="utf-8") as json_file:
                json.dump(outs, json_file, ensure_ascii=False, indent=4)