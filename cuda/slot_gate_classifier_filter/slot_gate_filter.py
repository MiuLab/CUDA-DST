"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import torch
from transformers import BertConfig, BertTokenizer
import json

try:
    from slot_gate_classifier_filter.run_filter import *
except:
    from run_filter import *
try:
    from slot_gate_classifier_filter.modeling import BertForMultiLabelValueClassification
except:
    from modeling import BertForMultiLabelValueClassification

class DataProcessor():
    def __init__(self,path):

        self.labels = {
            "Hotel":["Internet", "Parking", "Area", "Stars"],
            "Restaurant":["Area", "Food", "Price"],
            "Attraction":["Area", "Type"]
        }
        self.data = self.load_data(path)

    def __len__(self):
        return len(self.data)

    def load_data(self,path):
        multi_label_data = {}
        with open(path) as f:
            data = json.load(f)

            for dial_id, dial_data in data.items():
                history = ""
                system_transcript = ""

                for turn in dial_data["log"]:
                    if turn["turn_id"] % 2 == 0:
                        # user

                        label_list = []
                        multi_label_data[dial_id + str(turn["turn_id"]//2)] = {"text_a":system_transcript.strip(),
                                                                       "text_b": turn["text"].strip(),
                                                                       "label_list":label_list}
                    else:
                        # system
                        system_transcript = turn["text"].strip()    

        return multi_label_data

    def get_labels(self):
        """See base class."""
        bool_result = []
        dont_result = []
        result = []
        for domain, labels in self.labels.items():
            for label in labels:
                result.append(domain + "-" + label)
                if label in ["Internet", "Parking"]:
                    bool_result.append(domain + "-" + label)
                else:
                    dont_result.append(domain + "-" + label)
        return result, bool_result, dont_result
                
    def create_examples(self,dialogue_idx,turn_id,user_utters,user_act):
        examples = []
        meta_info = self.data[dialogue_idx+str(turn_id)]
        for user_utter in user_utters:
            text_a = meta_info["text_a"]
            text_b = user_utter

            label_list = {}
            for domain, slots in self.labels.items():
                for slot in slots:
                    label_list[domain + "-" + slot] = 0

            for domain, slots in user_act["Inform"].items():
                if domain in self.labels:
                    for label in slots:
                        if label[0] in self.labels[domain]:
                            flag = True
                            if label[1] in ["dontcare", "do n't care"]:
                                value = 1
                            elif label[1] == "no":
                                value = 2
                            elif label[1] in ["yes", "none", "free"]:
                                value = 3
                            else:
                                value = 2
                            label_list[domain + "-" + label[0]] = value
            # print("text_a: ",text_a.strip())
            # print("text_b: ",text_b.strip())
            # print("*************************")
            examples.append(InputExample(text_a=text_a.strip(),text_b = text_b.strip(),label=label_list))
            
        return examples



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None

        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
                
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=example.label))

    return features


def convert_examples_to_tensor(examples,label_list,max_seq_length,tokenizer):
    
    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)
    input_ids = []
    input_mask = []
    segment_ids = []
    all_label_ids = {}

    for f in features:
        input_ids.append(f.input_ids)
        input_mask.append(f.input_mask)
        segment_ids.append(f.segment_ids)           

    for s in features[0].label_id:
        all_label_ids[s] = torch.tensor([f.label_id[s] for f in features], dtype=torch.long)

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)

    data = (all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    
    return data

class BERTSlotGateFilter(object):

    def __init__(self,data_file):

        self.processor = DataProcessor(data_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_list, self.boolean_target_slot_list, self.dontcare_target_slot_list = self.processor.get_labels()
        bert_config = BertConfig.from_pretrained("bert-base-uncased",num_labels=len(self.label_list))
        bert_config.hidden_dropout_rate = 0.2
        bert_config.heads_dropout_rate = 0.0
        bert_config.boolean_target_slot_list = self.boolean_target_slot_list
        bert_config.dontcare_target_slot_list = self.dontcare_target_slot_list
        self.max_seq_length = 512
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.model = BertForMultiLabelValueClassification.from_pretrained("bert-base-uncased",config = bert_config)
#         import pdb;
#         pdb.set_trace();
#         import sys
        try:
            self.model.load_state_dict(torch.load("./slot_gate_classifier_filter/filter/best_model.pt", map_location='cpu'))
        except:
            self.model.load_state_dict(torch.load("./filter/best_model.pt", map_location='cpu'))
        self.model.to(self.device)

    def query_filter(self,dialogue_idx,turn_id,user_utters,turn_label,thresh):

        examples = self.processor.create_examples(dialogue_idx,turn_id,user_utters,turn_label)
        data = convert_examples_to_tensor(examples, self.label_list, self.max_seq_length, self.tokenizer)
        result = self.evaluation(data,thresh, turn_label)
        # print(result)
        return result

    def evaluation(self,data,thresh, turn_label):

        self.model.eval()
        prediction_list = []
        target_list = []
        input_ids, input_mask, segment_ids, label_ids  = data
        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)
        label_ids = {k: v.to(self.device) for k, v in label_ids.items()}
        with torch.no_grad():
            logits = self.model(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask)

        # print(logits)
        prediction_list,target_list = self.acc_pred(logits, label_ids, self.label_list,thresh, turn_label)
        # print(prediction_list)
        # print(target_list)
        result = []
        for idx in range(len(prediction_list)):
            flag = True
            for i in range(len(prediction_list[idx])):
                if prediction_list[idx][i]!=target_list[idx][i]:
                    flag = False
            if flag:
                result.append(True)
            else:
                result.append(False)
            # print("pred: ",prediction_set)
            # print("target: ",target_set)
            # print("*************************")
        # print(prediction_list)
        # print(target_list)
        return result

    def acc_pred(self,logits,labels,label_list,thresh, turn_label):
        batch_size = list(labels.values())[0].size(0)

        this_labels = []
        for domain, label in turn_label["Inform"].items():
            for slot, value in label:
                this_labels.append(domain + "-" + slot)

        prediction_list = []
        target_list = []
        for i in range(batch_size):
            prediction_list.append([])
            target_list.append([])
            
        for slot, value in labels.items():
            if slot in this_labels:
                for i in range(batch_size):
                    prediction_list[i].append(logits[slot][i].cpu().numpy())
                    target_list[i].append(value[i].cpu().numpy())
        prediction_list = np.asarray(prediction_list).astype(int)
        target_list = np.asarray(target_list).astype(int)

        return prediction_list,target_list

        

if __name__ == "__main__":

    # processor = DSTProcessor("../../MultiWOZ-coref/MultiWOZ2_3/data.json")



    classifier_filter = BERTSlotGateFilter("../../MultiWOZ-coref/MultiWOZ2_3/data.json")
    # while(True):
    dialogue_idx = "MUL2261.json"
    turn_id = 3
    thresh=0.5
    user_utters =[
        "<pad> I am also looking for a place to stay. It doesn't need to have internet.</s>",
        "<pad> I am also looking for a place to stay. I don't need internet.</s> <pad> <pad>",
        "<pad> I am also looking for a place to stay. It doesn't need to include internet.</s>",
        "<pad> I am also looking for a place to stay. Does it have internet?</s> <pad> <pad> <pad> <pad> <pad> <pad>",
        "<pad> I am also looking for a place to stay. I don't need free internet.</s> <pad>"
    ]
    turn_label = {
        "Inform": {
            "Hotel": [
                [
                    "Internet",
                    "no"
                ]
            ]
        },
        "Request": {},
        "Confirm": {}
    }
    flag = classifier_filter.query_filter(dialogue_idx,turn_id,user_utters,turn_label,thresh)
    print(flag)
    # import pdb;
    # pdb.set_trace()
    
 
