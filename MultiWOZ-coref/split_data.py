import os 
import json

data_path = "./MultiWOZ2_3/data.json"
train_list_file = "./trainListFile.txt"
test_list_file = "./testListFile.txt"
val_list_file = "./valListFile.txt"
train_file = "./MultiWOZ2_3/train.json"
test_file = "./MultiWOZ2_3/test.json"
val_file = "./MultiWOZ2_3/val.json"

with open(train_list_file, "r") as f:
    train_list = f.readlines()
    train_list = [i.strip('\n') for i in train_list]

with open(test_list_file, "r") as f:
    test_list = f.readlines()
    test_list = [i.strip('\n') for i in test_list]

with open(val_list_file, "r") as f:
    val_list = f.readlines()
    val_list = [i.strip('\n') for i in val_list]

train_data = {}
test_data = {}
val_data = {}

with open(data_path, "r") as f:
    data = json.load(f)

for id in train_list:
    if id in data:
        train_data[id] = data[id]

for id in test_list:
    if id in data:
        test_data[id] = data[id]

for id in val_list:
    if id in data:
        val_data[id] = data[id]

with open(train_file, "w") as f:
    json.dump(train_data, f, indent=4)

with open(test_file, "w") as f:
    json.dump(test_data, f, indent=4)

with open(val_file, "w") as f:
    json.dump(val_data, f, indent=4)
