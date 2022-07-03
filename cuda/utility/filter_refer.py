import os 
import json

data_path = "../cuda_data/cuda_out_cuda_classifier_binary_max-3_replyReqRate-0.9_confirmRate-0.7_newDomainRate-0.8_tryReferRate-0.6_bool_confirm_single_recommend_dontcare_seed_0.json"
target_path = "./filter_refer_data.json"

refer_data = {}

with open(data_path, "r") as f:
    all_data = json.load(f)

for id, data in all_data.items():
    flag = False
    if type(data) == type(dict()):
        for label in data["belief_state"]:
            if label["act"] == "coref":
                flag = True
                break
        
        if flag:
            refer_data[id] = data

with open(target_path, "w") as f:
    json.dump(refer_data, f, indent=4)

print(len(refer_data))
