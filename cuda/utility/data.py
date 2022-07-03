"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
# from utility.fix_label import fix_general_label_error
import numpy as np
import random

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi", "general", "booking"]
DOMAIN_LIST = ["Train", "Hotel", "Taxi", "Attraction", "Restaurant"]
SLOT_MAPS = {
    "pricerange": "price range",
    "arriveBy": "arrive by",
    "leaveAt": "leave at"
}
ONTOLOGY = {
    "attraction":set(["area", "name", "type"]),
    "hotel": set(["area","book day","book people", "book stay", "internet", "name","parking", "price range", "stars","type"]),
    "restaurant": set(["area","book day","book people","book time","food","name","price range"]),
    "taxi": set(["arrive by","departure", "destination", "leave at"]),
    "train":set(["arrive by","book people","day","departure","destination","leave at"])
}
TYPE_KEY_DICT = {
    "Booking-Inform": "Request_Book",
    "Train-OfferBook": "Request_Book", 
    "Train-OfferBooked": "Book", 
    "Booking-Book": "Book",
    "general-reqmore": "reqmore"
}
SLOT_METADATA_MAPS = {
    "Area": "area",
    "Arrive": "arriveBy",
    "Day": "day",
    "Dest": "destination",
    "Depart": "departure",
    "Food": "food",
    "Internet": "internet",
    "Leave": "leaveAt",
    "Name": "name",
    "Parking": "parking",
    "Price": "pricerange",
    "Stars": "stars",
    "Type": "type"
}

class MultiWOZForT5(Dataset):

    def __init__(self, max_src_len , max_tgt_len ,data_dir, tokenizer, shuffle_turn_label):

        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.shuffle_turn_label = shuffle_turn_label
        self.data = self.load_data(data_dir)

    def __getitem__(self, idx):

        data_detail = self.data[idx]
        history = data_detail["history"].strip()
        user = data_detail["text"].strip()
        user_act = data_detail["user_act"]
        system_act = data_detail["system_act"]
        
        user_act_str = self.get_user_act_string_withoutReq(user_act)
        system_act_str = self.get_system_act_string(system_act)

        src_text = history + " [SEP] "+ user_act_str + " [SEP] "+ system_act_str + " </s>"
        tgt_text = user + " </s>"
        inputs = self.tokenizer.encode_plus(src_text, pad_to_max_length=True,truncation=True, max_length=self.max_src_len)
        targets = self.tokenizer.encode_plus(tgt_text, pad_to_max_length=True,truncation=True, max_length=self.max_tgt_len)

        return {"src_ids": inputs["input_ids"],
                "src_mask":inputs["attention_mask"],
                "tgt_ids":targets["input_ids"],
                "tgt_mask":targets["attention_mask"]}

    def __len__(self):
        return len(self.data)
    
    def load_data(self,file_name):
        with open(file_name) as f:
            data = []
            dials = json.load(f)
            for dial_id, dial_data in dials.items():
                # check domain
                check_domain = True
                cur_domains = dial_data["new_goal"].keys()
                for current_domain in cur_domains:
                    if(current_domain not in EXPERIMENT_DOMAINS):
                        check_domain = False
                if not check_domain:
                    continue # We skip turns that doesn't appear in EXPERIMENT_DOMAINS

                # init system act
                system_act = {
                    "Inform": {}, 
                    "Request": {}, 
                    "Recommend": {}, 
                    "NoOffer": {}, 
                    "NoBook": {}, 
                    "Request_Book": [False, {}], 
                    "Book": False, 
                    "reqmore": False
                }
                confirm = {}
                history = ""
                context = "<system>: "
                domain = ""
                belief_state = []
                multi_recommend = False
                been_pass = False
                for turn in dial_data["log"]:
                    if turn["turn_id"] % 2 == 0:
                        # user
                        user_act, next_domain = self.get_user_act(turn["dialog_act"], domain)
                        user_act["Confirm"] = confirm
                        context = context + " <user>: " + turn["text"].strip()
                        data_detail = {
                            "text":turn["text"], 
                            "history": history,
                            "user_act": user_act,
                            "system_act": system_act,
                            "dialogue_id": dial_id,
                            "turn_id": turn["turn_id"]/2,
                            "context": context,
                            "pre_domain": domain,
                            "belief_state": belief_state,
                        }
                        domain = next_domain   
                        if not multi_recommend: 
                            data.append(data_detail)
                        else:
                            multi_recommend = False
                            been_pass = True
                        history = history + " [USR] " + turn["text"].strip()
                    else:
                        # system
                        system_act, domain = self.get_system_act(turn["dialog_act"], domain)
                        if (system_act["Recommend"] != {} and len(dial_data["log"]) > turn["turn_id"] + 2):
                            multi_recommend = self.check_multi_recommend(system_act["Recommend"])
                            confirm = self.check_bool_recommend(system_act["Recommend"], dial_data["log"][turn["turn_id"] + 2]["metadata"])
                        belief_state = self.get_belief_state(turn["metadata"], cur_domains)
                        if not been_pass:
                            data[-1]["original_belief_state"] = belief_state
                        else:
                            been_pass = False
                        history = history + " [SYS] " + turn["text"].strip()
                        context = context + " <system>: " + turn["text"].strip()
                    
        return data

    def get_user_act(self,dialogue_act, cur_domain):
        this_domain = cur_domain
        domain_slot_value_maps = {"Inform": {}, "Request": {}, "Confirm": False}
        for domain_type, slot_value_list in dialogue_act.items():
            domain,type = domain_type.split("-")
            if (domain in DOMAIN_LIST and domain != cur_domain):
                this_domain = domain
            for slot_value in slot_value_list:
                slot = slot_value[0]
                value = slot_value[1]
                if(value=="none"):
                    continue

                # check type 
                if(type == "Inform"):
                    if(domain not in domain_slot_value_maps[type]):
                        domain_slot_value_maps[type][domain] = [[slot,value]]
                    else:
                        domain_slot_value_maps[type][domain].append([slot,value])
                elif(type == "Request"):
                    if(domain not in domain_slot_value_maps[type]):
                        domain_slot_value_maps[type][domain] = [slot]
                    else:
                        domain_slot_value_maps[type][domain].append(slot)

                
        return domain_slot_value_maps, this_domain

    def get_user_act_string(self, user_act):
        result = ""
        inform_list = []
        for domain , slots in user_act["Inform"].items():
            for slot , value in slots:
                inform_list.append(domain + " " + slot + " " + value)
        if(self.shuffle_turn_label):
            random.shuffle(inform_list)
        if len(inform_list) != 0:
            result += " <INFORM> " + " , ".join(inform_list)
        
        request_list = []
        for domain , slots in user_act["Request"].items():
            for slot in slots:
                request_list.append(domain + " " + slot)
        if(self.shuffle_turn_label):
            random.shuffle(request_list)
        if len(request_list) != 0:
            result += " <REQUEST> " + " , ".join(request_list)

        # confirm_list = []
        # for domain , slots in user_act["Confirm"].items():
        #     for slot , value in slots:
        #         confirm_list.append(domain + " " + slot + " " + value)
        # if(self.shuffle_turn_label):
        #     random.shuffle(confirm_list)
        # if len(confirm_list) != 0:
        #     result += " <CONFIRM> " + " , ".join(confirm_list)
        if user_act["Confirm"]:
            result += " <CONFIRM> "

        return result

    def get_user_act_string_withoutReq(self, user_act):
        result = ""
        inform_list = []
        for domain , slots in user_act["Inform"].items():
            for slot , value in slots:
                inform_list.append(domain + " " + slot + " " + value)
        if(self.shuffle_turn_label):
            random.shuffle(inform_list)
        if len(inform_list) != 0:
            result += " <INFORM> " + " , ".join(inform_list)

        # confirm_list = []
        # for domain , slots in user_act["Confirm"].items():
        #     for slot , value in slots:
        #         confirm_list.append(domain + " " + slot + " " + value)
        # if(self.shuffle_turn_label):
        #     random.shuffle(confirm_list)
        # if len(confirm_list) != 0:
        #     result += " <CONFIRM> " + " , ".join(confirm_list)
        if user_act["Confirm"]:
            result += " <CONFIRM> "

        return result

    def get_system_act(self,dialogue_act, cur_domain):

        this_domain = cur_domain
        domain_slot_value_maps = {
            "Inform": {}, 
            "Request": {}, 
            "Recommend": {}, 
            "NoOffer": {}, 
            "NoBook": {}, 
            "Request_Book": [False, {}], 
            "Book": False, 
            "reqmore": False
        }
        for domain_type, slot_value_list in dialogue_act.items():
            domain,type = self.get_domain_type(domain_type)
            if (domain in DOMAIN_LIST and domain != cur_domain):
                this_domain = domain
            for slot_value in slot_value_list:
                slot = slot_value[0]
                value = slot_value[1]
                
                if(type == "reqmore"):
                    domain_slot_value_maps[type] = True
                elif(domain_type in ["Booking-Book", "train-OfferBooked"]):
                    domain_slot_value_maps[type] = True
                elif(domain_type in ["Booking-Inform","Train-OfferBook"]):
                    domain_slot_value_maps[type][0] = True
                    if(slot != "none"):
                        domain_slot_value_maps[type][1][slot] = value
                elif(type in ["Inform", "Recommend", "NoOffer"]):
                    if(domain not in domain_slot_value_maps[type]):
                        domain_slot_value_maps[type][domain] = [[slot,value]]
                    else:
                        domain_slot_value_maps[type][domain].append([slot,value])
                elif(type == "Request"):
                    if(domain not in domain_slot_value_maps[type]):
                        domain_slot_value_maps[type][domain] = [slot]
                    else:
                        domain_slot_value_maps[type][domain].append(slot)
                elif(type == "NoBook"):
                    domain_slot_value_maps[type][slot] = value
            
                
        return domain_slot_value_maps, this_domain

    def get_system_act_string(self, sys_act):
        result = ""
        inform_list = []
        for domain , slots in sys_act["Inform"].items():
            for slot , value in slots:
                inform_list.append(domain + " " + slot + " " + value)
        if(self.shuffle_turn_label):
            random.shuffle(inform_list)
        if len(inform_list) != 0:
            result += " <INFORM> " + " , ".join(inform_list)

        recommend_list = []
        for domain , slots in sys_act["Recommend"].items():
            for slot , value in slots:
                recommend_list.append(domain + " " + slot + " " + value)
        if(self.shuffle_turn_label):
            random.shuffle(recommend_list)
        if len(recommend_list) != 0:
            result += " <RECOMMEND> " + " , ".join(recommend_list)

        nooffer_list = []
        for domain , slots in sys_act["NoOffer"].items():
            for slot , value in slots:
                nooffer_list.append(domain + " " + slot + " " + value)
        if(self.shuffle_turn_label):
            random.shuffle(nooffer_list)
        if len(nooffer_list) != 0:
            result += " <NOOFFER> " + " , ".join(nooffer_list)

        nobook_list = []
        for slot, value in sys_act["NoBook"].items():
            nobook_list.append(slot + " " + value)
        if(self.shuffle_turn_label):
            random.shuffle(nobook_list)
        if len(nobook_list) != 0:
            result += " <NOBOOK> " + " , ".join(nobook_list)
        
        request_list = []
        for domain , slots in sys_act["Request"].items():
            for slot in slots:
                request_list.append(domain + " " + slot)
        if(self.shuffle_turn_label):
            random.shuffle(request_list)
        if len(request_list) != 0:
            result += " <REQUEST> " + " , ".join(request_list)

        if sys_act["Request_Book"][0]:
            result += " <REQUEST_BOOK> "
            reqbook_list = []
            for slot, value in sys_act["Request_Book"][1].items():
                reqbook_list.append(slot + " " + value)
            if(self.shuffle_turn_label):
                random.shuffle(reqbook_list)
            if len(reqbook_list) != 0:
                result += " , ".join(reqbook_list)

        if sys_act["Book"]:
            result += " <BOOK> "
        
        if sys_act["reqmore"]:
            result += " <REQMORE> "

        return result

    def get_domain_type(self, domain_type):
        domain, type = domain_type.split('-')
        if domain_type in TYPE_KEY_DICT:
            type = TYPE_KEY_DICT[domain_type]
        return domain, type

    def get_belief_state(self, metadata, domains):
        belief_state = []
        for domain in domains:
            for slot, value in metadata[domain]["book"].items():
                if slot != "booked":
                    if value not in ["", "not mentioned"]:
                        belief_state.append([domain + "-" + slot, value])
            for slot, value in metadata[domain]["semi"].items():
                if value not in ["", "not mentioned"]:
                    belief_state.append([domain + "-" + slot, value])
        
        return belief_state

    def check_recommend(self, recommends, metadata):
        confirm = {}
        for domain, slots in recommends.items():
            for slot, value in slots:
                if slot in SLOT_METADATA_MAPS:
                    if slot in ["Parking", "Internet"]:
                        if (value in ["yes", "free", "none"] and metadata[domain.lower()]["semi"][SLOT_METADATA_MAPS[slot]] == "yes") or (value == metadata[domain.lower()]["semi"][SLOT_METADATA_MAPS[slot]]):
                            if domain not in confirm:
                                confirm[domain] = [[slot, value]]
                            else:
                                confirm[domain].append([slot, value])
                        
                    # elif "|" in metadata[domain.lower()]["semi"][SLOT_METADATA_MAPS[slot]]:
                    #     break
                    else:
                        meta_value = metadata[domain.lower()]["semi"][SLOT_METADATA_MAPS[slot]]
                        value_list = meta_value.split("|")
                        if (value.lower().replace(" 's ", "s ").strip() in value_list or value.lower().replace(" 's ", " ").strip() in value_list):
                            if domain not in confirm:
                                confirm[domain] = [[slot, value]]
                            else:
                                confirm[domain].append([slot, value])
                        else:
                            if "centre" in value_list:
                                value_list.append("center")
                            for meta in value_list:
                                if meta in value.lower():
                                    if domain not in confirm:
                                        confirm[domain] = [[slot, value]]
                                    else:
                                        confirm[domain].append([slot, value])
        return confirm
    
    def check_bool_recommend(self, recommends, metadata):
        confirm = {}
        for domain, slots in recommends.items():
            for slot, value in slots:
                if slot in SLOT_METADATA_MAPS:
                    if slot in ["Parking", "Internet"]:
                        if (value in ["yes", "free", "none"] and metadata[domain.lower()]["semi"][SLOT_METADATA_MAPS[slot]] == "yes") or (value == metadata[domain.lower()]["semi"][SLOT_METADATA_MAPS[slot]]):
                            if domain not in confirm:
                                confirm[domain] = [slot]
                            else:
                                confirm[domain].append(slot)
                    else:
                        meta_value = metadata[domain.lower()]["semi"][SLOT_METADATA_MAPS[slot]]
                        value_list = meta_value.split("|")
                        if (value.lower().replace(" 's ", "s ").strip() in value_list or value.lower().replace(" 's ", " ").strip() in value_list):
                            if domain not in confirm:
                                confirm[domain] = [slot]
                            else:
                                confirm[domain].append(slot)
                        else:
                            if "centre" in value_list:
                                value_list.append("center")
                            for meta in value_list:
                                if meta in value.lower():
                                    if domain not in confirm:
                                        confirm[domain] = [slot]
                                    else:
                                        confirm[domain].append(slot)

        return confirm != {}
                    
    def check_multi_recommend(self, recommends):
        for labels in recommends.values():
            exist = []
            for slot_value in labels:
                if slot_value[0] in exist:
                    return True
                else:
                    exist.append(slot_value[0])
        return False
    


class MultiWOZForT5_Interact(MultiWOZForT5):

    prompt_dict = {}

    def __init__(self, data_dir, tokenizer, shuffle_turn_label):
        self.tokenizer = tokenizer
        self.shuffle_turn_label = shuffle_turn_label
        self.data = self.load_data(data_dir)

    def __getitem__(self, idx):
        data_detail = self.data[idx]
        history = data_detail["history"].strip()  
        system_act = data_detail["system_act"]

        user_act_str = self.get_user_act_string_withoutReq(self.prompt_dict)
        system_act_str = self.get_system_act_string(system_act)

        src_text = history + " [SEP] "+ user_act_str + " [SEP] "+ system_act_str + " </s>"

        input_ids = self.tokenizer(src_text)["input_ids"]

        return input_ids

    def get_dialID_turnID(self, idx):
        data_detail = self.data[idx]
        dialogue_idx = data_detail["dialogue_id"].strip()
        turn_idx = int(data_detail["turn_id"])
        return dialogue_idx, turn_idx

    def print_value(self, idx):
        data_detail = self.data[idx]
        sys = data_detail["system"].strip()
        user = data_detail["user"].strip()
        domain_slot_value_maps = data_detail["domain_slot_value_maps"]
        domain_slot_value_list = []
        for key, values in domain_slot_value_maps.items():
            for name, value in values:
                if value != "none":
                    domain_slot_value_list.append(key + "-"+name + "-" + value)

        domain_slot_value_str = " , ".join(domain_slot_value_list)

        print("==> Original SYS Utterance: ", sys)
        print("==> Original Turn-level BS: ", domain_slot_value_str)
        print("==> Original USR Utterance: ", user)


def get_dataloader(dataset, tokenizer, args, split='train'):

    def T5collate_fn(batch):
        """
        Modify target_id as label, T5 will modify label as valid taget input and add bos token
        """
        src_ids = torch.tensor([example['src_ids'] for example in batch], dtype=torch.long)
        src_mask = torch.tensor([example['src_mask'] for example in batch], dtype=torch.long)
        tgt_ids = torch.tensor([example['tgt_ids'] for example in batch], dtype=torch.long)
        tgt_ids[tgt_ids[:, :] == 0] = -100
        tgt_mask = torch.tensor([example['tgt_mask'] for example in batch], dtype=torch.long)
        
        return {"src_ids": src_ids,
                "src_mask":src_mask,
                "tgt_ids":tgt_ids,
                "tgt_mask":tgt_mask}

    if split == 'train':
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        batch_size = args.train_batch_size
        sampler = RandomSampler(dataset if args.local_rank == -1 else DistributedSampler(dataset)) # SequentialSampler(dataset)
    else:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        batch_size = args.eval_batch_size
        sampler = SequentialSampler(dataset)
    
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=T5collate_fn)

    return dataloader, args
