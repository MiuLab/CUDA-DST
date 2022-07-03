"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import numpy as np
import copy
import random
import re 

DOMAIN_LIST = ["Train", "Hotel", "Taxi", "Attraction", "Restaurant"]
SLOT_BELIEF_MAPS = {
    "Hotel":{
        "Internet": "hotel-internet", 
        "Type": "hotel-type", 
        "Parking": "hotel-parking", 
        "Price": "hotel-pricerange", 
        "Day": "hotel-book day", 
        "People": "hotel-book people", 
        "Stay": "hotel-book stay", 
        "Area": "hotel-area", 
        "Stars": "hotel-stars", 
        "Name": "hotel-name"
    },
    "Restaurant":{
        "Area": "restaurant-area", 
        "Food": "restaurant-food", 
        "Price": "restaurant-pricerange", 
        "Name": "restaurant-name", 
        "Day": "restaurant-book day", 
        "People": "restaurant-book people", 
        "Time": "restaurant-book time"
    },
    "Taxi":{
        "Arrive": "taxi-arriveby", 
        "Leave": "taxi-leaveat", 
        "Depart": "taxi-departure", 
        "Dest": "taxi-destination"
    },
    "Train":{
        "Arrive": "train-arriveby", 
        "Leave": "train-leaveat", 
        "Depart": "train-departure", 
        "Dest": "train-destination", 
        "Day": "train-day", 
        "People": "train-book people"
    },
    "Attraction":{
        "Area": "attraction-area", 
        "Name": "attraction-name", 
        "Type": "attraction-type"
    }
}
MULTIWOZ_BELIEF_SLOT_MAPS = {
    "hotel-internet": "hotel-internet", 
    "hotel-type": "hotel-type", 
    "hotel-parking": "hotel-parking", 
    "hotel-pricerange": "hotel-pricerange", 
    "hotel-day": "hotel-book day", 
    "hotel-people": "hotel-book people", 
    "hotel-stay": "hotel-book stay", 
    "hotel-area": "hotel-area", 
    "hotel-stars": "hotel-stars", 
    "hotel-name": "hotel-name",

    "restaurant-area": "restaurant-area", 
    "restaurant-food": "restaurant-food", 
    "restaurant-pricerange": "restaurant-pricerange", 
    "restaurant-name": "restaurant-name", 
    "restaurant-day": "restaurant-book day", 
    "restaurant-people": "restaurant-book people", 
    "restaurant-time": "restaurant-book time",

    "taxi-arriveBy": "taxi-arriveby", 
    "taxi-leaveAt": "taxi-leaveat", 
    "taxi-departure": "taxi-departure", 
    "taxi-destination": "taxi-destination",

    "train-arriveBy": "train-arriveby", 
    "train-leaveAt": "train-leaveat", 
    "train-departure": "train-departure", 
    "train-destination": "train-destination", 
    "train-day": "train-day", 
    "train-people": "train-book people",

    "attraction-area": "attraction-area", 
    "attraction-name": "attraction-name", 
    "attraction-type": "attraction-type"
}
REFER_LIST_DICT = {
    "Hotel":{
        "Price": {'restaurant-pricerange':["same", "same price", "same price range"]} ,
        "Day": {'train-day': ["same", "same day"], 'restaurant-day': ["same", "same day"]},
        "People": {'train-people': ["same", "same group", "same party"], 'restaurant-people': ["same", "same group", "same party"]},
        "Area": {'restaurant-area': ["same", "same area", "same part", "near the restaurant"], 'attraction-area':["same", "same area", "same part", "near the attraction"]},
    },
    "Restaurant":{
        "Area": {'hotel-area': ["same", "same area", "same part", "near the hotel"], 'attraction-area':["same", "same area", "same part", "near the attraction"]},
        "Price": {'hotel-pricerange':["same", "same price", "same price range"]} ,
        "Day": {'train-day': ["same", "same day"], 'hotel-day': ["same", "same day"]},
        "People": {'train-people': ["same", "same group", "same party"], 'hotel-people': ["same", "same group", "same party"]},
    },
    "Taxi":{
        "Depart": {'hotel-name':["the hotel"], 'restaurant-name':["the restaurant"], 'attraction-name':["the attraction"]},
        "Dest": {'hotel-name':["the hotel"], 'restaurant-name':["the restaurant"], 'attraction-name':["the attraction"]},
        "Arrive": {'restaurant-time': ["the time of my reservation", "the time of my booking"]}
    },
    "Train":{
        "Day":{'restaurant-day': ["same", "same day"], 'hotel-day': ["same", "same day"]},
        "People":{'restaurant-people': ["same", "same group", "same party"], 'hotel-people': ["same", "same group", "same party"]},
    },
    "Attraction":{
        "Area": {'hotel-area': ["same", "same area", "same part", "near the hotel"], 'restaurant-area':["same", "same area", "same part", "near the restaurant"]},
    }
}

def re_match(utter,value):
    search_span = re.search(r'[?,.! ]'+value+r'[?,.! ]'," "+utter+" ")
    if(search_span):
        return True
    else:
        return False

def update_turn(turn,new_turn_label_dict):

    old_turn_label_dict = {}
    for (slot,value) in turn["turn_label"]:
        old_turn_label_dict[slot] = value

    new_belief = []
    slot_set = set()
    for bs in turn["belief_state"]:
        slot , value = bs["slots"][0]
        copy_bs = copy.deepcopy(bs)
        slot_set.add(slot)                                                 # Record Slot in Previous Turns
        if(slot in old_turn_label_dict and slot in new_turn_label_dict):   # Update Slot Value in Current Turns
            new_value = new_turn_label_dict[slot]
            copy_bs["slots"][0][1] = new_value
            if(new_value in turn["system_transcript"]):                     # TODO. This can be done by compare old_value with new_value
                copy_bs["act"] = "keep"
            else:
                copy_bs["act"] = "update"
            new_belief.append(copy_bs)

        elif(slot not in old_turn_label_dict):                              # Maintain Slot,value in previous turn
            copy_bs["act"] = "keep"
            new_belief.append(copy_bs)                  

    new_turn_label = []
    for slot,value in new_turn_label_dict.items():
        if(slot not in slot_set):                                           # New Added Slot,value in should not appear in current turn's bs.
            copy_bs = {"slots": [[slot,value]],"act": "add"}
            new_belief.append(copy_bs)

        new_turn_label.append([slot,value])


    turn["belief_state"] = new_belief
    turn["turn_label"] = new_turn_label
    return turn

def gen_time_pair():

    time_formats = ["am","pm","standard"]
    time_format = np.random.choice(time_formats,1)[0]
    if(time_format=="am" or time_format=="pm"):
        hour = random.randint(1,11)
        leave_min = random.randint(10,29)
        arrive_min = leave_min + random.randint(10,30)
        leave_time = str(hour)+":"+str(leave_min)+" "+time_format
        arrive_time = str(hour)+":"+str(arrive_min)+" "+time_format
    else:
        hour = random.randint(13,23)
        leave_min = random.randint(10,29)
        arrive_min = leave_min + random.randint(10,30)
        leave_time = str(hour)+":"+str(leave_min)
        arrive_time = str(hour)+":"+str(arrive_min)

    return(leave_time,arrive_time)

def fix_commonsense(user_act):

    if("Taxi" in user_act["Inform"]):
        if(("Arrive" in user_act["Inform"]["Taxi"]) and ("Leave" in user_act["Inform"]["Taxi"])):
            leave_time,arrive_time = gen_time_pair()
            user_act["Inform"]["Taxi"]["Leave"] = leave_time
            user_act["Inform"]["Taxi"]["Arrive"] = arrive_time
    if("Train" in user_act["Inform"]):
        if(("Arrive" in user_act["Inform"]["Train"]) and ("Leave" in user_act["Inform"]["Train"])):
            leave_time,arrive_time = gen_time_pair()
            user_act["Inform"]["Train"]["Leave"] = leave_time
            user_act["Inform"]["Train"]["Arrive"] = arrive_time

    return user_act

def check_multi_recommend(recommends):
        for labels in recommends.values():
            exist = []
            for slot_value in labels:
                if slot_value[0] in exist:
                    return True
                else:
                    exist.append(slot_value[0])
        return False

def generate_new_turn_label(turn):
    new_turn_label_dict = {}
    turn_label_dict = {}
    for domain, slot_values in turn["system_act"]["Inform"].items():
        for slot, value in slot_values:
            if domain in SLOT_BELIEF_MAPS:
                if slot in SLOT_BELIEF_MAPS[domain]:
                    domain_slot = SLOT_BELIEF_MAPS[domain][slot]
                    new_turn_label_dict[domain_slot] = value
                    turn_label_dict[domain + "-" + slot] = value
    if turn["user_act"]["Confirm"]:
        for domain, slot_values in turn["system_act"]["Recommend"].items():
            for slot, value in slot_values:
                if domain in SLOT_BELIEF_MAPS:
                    if slot in SLOT_BELIEF_MAPS[domain]:
                        domain_slot = SLOT_BELIEF_MAPS[domain][slot]
                        new_turn_label_dict[domain_slot] = value
                        turn_label_dict[domain + "-" + slot] = value
    for domain, slot_values in turn["user_act"]["Inform"].items():
        for slot, value in slot_values:
            if domain in SLOT_BELIEF_MAPS:
                if slot in SLOT_BELIEF_MAPS[domain]:
                    domain_slot = SLOT_BELIEF_MAPS[domain][slot]
                    new_turn_label_dict[domain_slot] = value
                    turn_label_dict[domain + "-" + slot] = value
    
    new_turn_label = []
    turn_label = []
    for domain_slot, value in new_turn_label_dict.items():
        new_turn_label.append([domain_slot, value])
    for domain_slot, value in turn_label_dict.items():
        turn_label.append([domain_slot, value])

    turn["new_turn_label"] = new_turn_label
    turn["turn_label"] = turn_label
    return turn

def generate_origin_turn_label(turn):
    origin_belief_dict = {}
    turn_label = []
    for slot, value in turn["belief_state"]:
        origin_belief_dict[slot] = value
    for slot, value in turn["original_belief_state"]:
        if slot not in origin_belief_dict:
            turn_label.append([slot, value])
        elif value != origin_belief_dict[slot]:
            turn_label.append([slot, value])

    turn["new_turn_label"] = turn_label
    return turn

def modify_belief_state_domain_slot(turn):
    for i, (slot, value) in enumerate(turn["original_belief_state"]):
        if slot in ["hotel-internet", "hotel-parking"] and value in ["yes", "none", "free"]:
            value = "yes"
        turn["original_belief_state"][i] = [MULTIWOZ_BELIEF_SLOT_MAPS[slot], value]
    for i, (slot, value) in enumerate(turn["belief_state"]):
        if slot in ["hotel-internet", "hotel-parking"] and value in ["yes", "none", "free"]:
            value = "yes"
        turn["belief_state"][i] = [MULTIWOZ_BELIEF_SLOT_MAPS[slot], value]
    return turn

def update_turn_label_with_coreference(turn):
    for tar_slot, tar_value in turn["coreference"].items():
        for i, (slot, value) in enumerate(turn["new_turn_label"]):
            if slot == tar_slot:
                turn["new_turn_label"][i][1] = tar_value
                break
    return turn

def update_belief_state_with_new_turn_label(turn):
    turn = update_turn_label_with_coreference(turn)
    belief_dict = {}
    updated_slot = []
    for slot, value in turn["belief_state"]:
        belief_dict[slot] = value
    for slot, value in turn["new_turn_label"]:
        belief_dict[slot] = value
        updated_slot.append(slot)
    new_belief_state = []
    for slot, value in belief_dict.items():
        if slot in turn["coreference"]:
            act = "coref"
        elif slot in updated_slot:
            act = "update"
        else:
            act = "keep"
        new_belief_state.append({
            "slots": [[slot, value]],
            "act": act
        })
    turn["belief_state"] = new_belief_state
    return turn

def update_belief_state(turn):
    new_belief_state = []
    for slot, value in turn["original_belief_state"]:
        new_belief_state.append({
            "slots": [[slot, value]],
            "act": "keep"
        })
    turn["belief_state"] = new_belief_state
    return turn

def counterfactual_goal_generator(turn, try_refer_rate, max_refer, reply_req_rate, confirm_rate, new_domain_rate, slot_value_dict, immutable_inform=True):
    user_act = {"Inform": {},"Request": {} , "Confirm": False}
    added_num = 0
    user_utter = turn["text"]
    system_act = turn["system_act"]
    domain = turn["pre_domain"]
    if domain == "":
        domain = random.choice(DOMAIN_LIST)


    immutable_slot = {"Train": [], "Hotel": [], "Taxi": [], "Attraction": [], "Restaurant": []}
    if system_act["Recommend"] != {}:
        if random.random() < confirm_rate:
            if check_multi_recommend(system_act["Recommend"]):
                user_act["Confirm"] = False
            else:
                user_act["Confirm"] = True
                for confirm_domain, slot_values in system_act["Recommend"].items():
                    if confirm_domain in immutable_slot and immutable_inform:
                        for slot, value in slot_values:
                            immutable_slot[confirm_domain].append(slot)
    
    inform_num = random.choice([1,1,2,2,2,3]) # don't add too many slot
    if(not system_act["Request_Book"][0] and system_act["NoOffer"] == {} and system_act["NoBook"] == {} and  system_act["Request"] == {}):
        if random.random() < new_domain_rate:
            domain = random.choice(DOMAIN_LIST)
    
        user_act["Inform"][domain] = []
        if len(list(set(slot_value_dict[domain].keys()) - set(immutable_slot[domain]))) >= inform_num:
            for slot in random.sample(list(set(slot_value_dict[domain].keys()) - set(immutable_slot[domain])), inform_num):
                user_act["Inform"][domain].append([slot, random.choice(slot_value_dict[domain][slot])])
        else:
            for slot in list(set(slot_value_dict[domain].keys()) - set(immutable_slot[domain])):
                user_act["Inform"][domain].append([slot, random.choice(slot_value_dict[domain][slot])])
        
    elif system_act["Request"] != {}:
        num = 0
        for req_domain, slots in system_act["Request"].items():
            if req_domain in DOMAIN_LIST:
                for slot in slots:
                    if slot in slot_value_dict[req_domain] and random.random() < reply_req_rate:
                        if req_domain not in user_act["Inform"]:
                            user_act["Inform"][req_domain] = [[slot,  random.choice(slot_value_dict[req_domain][slot])]]
                        else:
                            user_act["Inform"][req_domain].append([slot,  random.choice(slot_value_dict[req_domain][slot])])
                        num += 1
                        immutable_slot[req_domain].append(slot)

        inform_num -= num
        if(inform_num > 0):
            if domain not in user_act["Inform"]:
                user_act["Inform"][domain] = []
            if len(list(set(slot_value_dict[domain].keys()) - set(immutable_slot[domain]))) >= inform_num:
                for slot in random.sample(list(set(slot_value_dict[domain].keys()) - set(immutable_slot[domain])), inform_num):
                    user_act["Inform"][domain].append([slot, random.choice(slot_value_dict[domain][slot])])
            else:
                for slot in list(set(slot_value_dict[domain].keys()) - set(immutable_slot[domain])):
                    user_act["Inform"][domain].append([slot, random.choice(slot_value_dict[domain][slot])])
    else:
        user_act["Inform"][domain] = []
        if len(list(set(slot_value_dict[domain].keys()) - set(immutable_slot[domain]))) >= inform_num:
            for slot in random.sample(list(set(slot_value_dict[domain].keys()) - set(immutable_slot[domain])), inform_num):
                user_act["Inform"][domain].append([slot, random.choice(slot_value_dict[domain][slot])])
        else:
            for slot in list(set(slot_value_dict[domain].keys()) - set(immutable_slot[domain])):
                user_act["Inform"][domain].append([slot, random.choice(slot_value_dict[domain][slot])])


    user_act = fix_commonsense(user_act)

    turn["user_act"] = user_act

    refer = False
    turn["coreference"] = {}
    if random.random() < try_refer_rate:
        turn, count = add_coreference(turn, max_refer)
        refer = count

    turn = generate_new_turn_label(turn)

    return turn, refer

def add_coreference(turn, max):
    exist_label = {}
    coref_list = {}
    target_list = []
    count = 0
    for slot, value in turn["belief_state"]:
        exist_label[slot] = value
    for domain, slot_values in turn["user_act"]["Inform"].items():
        for i, (slot, value) in enumerate(slot_values):
            if slot in REFER_LIST_DICT[domain]:
                candidate = []
                for target, coref_str_list in REFER_LIST_DICT[domain][slot].items():
                    if target in exist_label and target not in target_list:
                        candidate.append([exist_label[target], random.choice(coref_str_list), target])
                if len(candidate) > 0:
                    apply = random.choice(candidate)
                    coref_list[SLOT_BELIEF_MAPS[domain][slot]] = apply[0]
                    turn["user_act"]["Inform"][domain][i][1] = apply[1]
                    target_list.append(apply[2])
                    count += 1
                    if count >= max:
                        break
        if count >= max:
            break
    
    turn["coreference"] = coref_list

    return turn, count
