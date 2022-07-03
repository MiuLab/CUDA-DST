from scipy import stats
import os
import json

target_dir = "../../coco-dst/coco_eval_noIgnore"
target_dir2 = "../../all_data/size1_seed0/eval/trippy/"
trippy_path = os.path.join(target_dir, "baseline/trade/coco-vs_rare_out_domain_test_classifier_change_add-2-max-3_drop-1_seed_0.json")
coco_path = os.path.join(target_dir, "coco/trade/coco-vs_rare_out_domain_test_classifier_change_add-2-max-3_drop-1_seed_0.json")
cuda_path = os.path.join(target_dir, "cuda/trade/coco-vs_rare_out_domain_test_classifier_change_add-2-max-3_drop-1_seed_0.json")
cuda_withoutRef_path = os.path.join(target_dir, "cuda_withoutRef/trade/coco-vs_rare_out_domain_test_classifier_change_add-2-max-3_drop-1_seed_0.json")

target_file = "./t-test_trade.json"

model_name = "trade"

id_list = []
all_data = {"multiWOZ": {}, "coco": {}, "result": {"multiWOZ": {}, "coco": {}}}
with open(trippy_path, "r") as f:
    data = json.load(f)
    id_list = data.keys()
    multiWOZ = []
    coco = []
    for id in id_list:
        if type(data[id]) == type(dict()):
            multiWOZ.append(data[id]["ori_prediction"]["Joint Acc"])
            coco.append(data[id]["new_prediction"]["Joint Acc"])
    all_data["multiWOZ"][model_name] = coco
    all_data["coco"][model_name] = multiWOZ

with open(coco_path, "r") as f:
    data = json.load(f)
    id_list = data.keys()
    multiWOZ = []
    coco = []
    for id in id_list:
        if type(data[id]) == type(dict()):
            multiWOZ.append(data[id]["ori_prediction"]["Joint Acc"])
            coco.append(data[id]["new_prediction"]["Joint Acc"])
    all_data["multiWOZ"]["coco"] = coco
    all_data["coco"]["coco"] = multiWOZ

with open(cuda_path, "r") as f:
    data = json.load(f)
    id_list = data.keys()
    multiWOZ = []
    coco = []
    for id in id_list:
        if type(data[id]) == type(dict()):
            multiWOZ.append(data[id]["ori_prediction"]["Joint Acc"])
            coco.append(data[id]["new_prediction"]["Joint Acc"])
    all_data["multiWOZ"]["cuda"] = coco
    all_data["coco"]["cuda"] = multiWOZ

with open(cuda_withoutRef_path, "r") as f:
    data = json.load(f)
    id_list = data.keys()
    multiWOZ = []
    coco = []
    for id in id_list:
        if type(data[id]) == type(dict()):
            multiWOZ.append(data[id]["ori_prediction"]["Joint Acc"])
            coco.append(data[id]["new_prediction"]["Joint Acc"])
    all_data["multiWOZ"]["cuda_withoutRef"] = multiWOZ
    all_data["coco"]["cuda_withoutRef"] = coco

target = all_data["multiWOZ"]
result = {}
for a in target:
    for b in target:
        if a != b:
            tStat, pValue = stats.ttest_rel(target[a], target[b])
            result[a + "-" + b] = {}
            result[a + "-" + b]["tStat"] = tStat
            result[a + "-" + b]["pValue"] = pValue
all_data["result"]["multiWOZ"] = result

target = all_data["coco"]
result = {}
for a in target:
    for b in target:
        if a != b:
            tStat, pValue = stats.ttest_rel(target[a], target[b])
            result[a + "-" + b] = {}
            result[a + "-" + b]["tStat"] = tStat
            result[a + "-" + b]["pValue"] = pValue
all_data["result"]["coco"] = result

with open(target_file, "w") as f:
    json.dump(all_data, f, indent=4)
