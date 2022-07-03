"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys
import logging
import numpy as np
import torch
from tqdm import tqdm
import re
import random
import copy
from utility.slot_value_ctrl import counterfactual_goal_generator, generate_origin_turn_label, modify_belief_state_domain_slot, update_belief_state_with_new_turn_label, update_belief_state
from utility.data import *
from utility.dictionary import *
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
)
import json

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "t5": (T5ForConditionalGeneration, T5Tokenizer)
}


def set_seed(args_seed, args_n_gpu):
    np.random.seed(args_seed)
    torch.manual_seed(args_seed)
    random.seed(args_seed)
    if args_n_gpu > 0:
        torch.cuda.manual_seed_all(args_seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def build_dict(file_test):
    data_maps = {}
    with open(file_test) as f:
        data = json.load(f)
        for id, dial in data.items():
            data_maps[id] = dial["new_goal"].keys()

    return data_maps


def dial_print(dial):
    print("Dial domains: ", dial['domains'])
    print("-------------")
    for idx, turn in enumerate(dial['dialogue']):
        print("Turn ID: ", idx, " Turn Domain: ", turn["domain"])
        print("Bot: ", turn["system_transcript"])
        print("User: ", turn["transcript"])
        print("-------------")


def get_context(dial, turn_id):
    context = ""
    for idx, turn in enumerate(dial['dialogue']):
        if idx <= turn_id:
            context += (" <system>: " + turn["system_transcript"] + " <user>: " + turn["transcript"])
        else:
            break

    return context


def turn_label2string(turn_label):
    label_list = []
    for (domain_slot, value) in turn_label:
        label_list.append(domain_slot + "-" + value)

    return " , ".join(label_list)


def para_filtering(turn, sentences, K):
    """filter paraphrase"""

    missing_values = []
    for domain_slot, value in turn["turn_label"]:
        if (value in turn["system_transcript"]) and (value not in turn["transcript"]):
            missing_values.append(value)

    value_list = []
    best_sent = ""
    for domain_slot, value in turn["turn_label"]:

        domain, slot = domain_slot.split("-")
        if slot == "parking":
            value = slot
        elif slot == "internet":
            value = "wifi"
        if value not in missing_values:
            value_list.append(value)

    count = 0
    for sent in sentences:
        sent = sent.lower()
        flag = True
        for value in value_list:
            if value not in sent:
                flag = False
                break

        if flag and (K == count):
            best_sent = sent
            break
        elif flag and (count < K):
            count += 1

    return best_sent


def match_filtering(new_turn, ori_turn, sentences):
    value_list = []
    best_sent = ""
    for slot_values in new_turn["user_act"]["Inform"].values():
        for slot, value in slot_values:
            if slot in ["Parking", "Internet"] or value == "dontcare":
                continue
            value_list.extend(value.lower().split())

    for sent in sentences:
        flag = True
        for value in value_list:
            if value not in sent.lower():
                flag = False
                break

        if flag:
            best_sent = sent
            break

    return best_sent


def classifier_filtering(classifier_filter, dialogue_idx, turn_id, sentences, turn_label, thresh):
    """Use bert to get qualified candidates"""

    qual_sents = []
    flags = classifier_filter.query_filter(dialogue_idx, turn_id, sentences, turn_label, thresh)
    for idx, flag in enumerate(flags):
        if flag:
            qual_sents.append(sentences[idx])

    return qual_sents


def slot_gate_filtering(slot_gate_filter, dialogue_idx, turn_id, sentences, turn_label, thresh):
    """Use bert to get qualified candidates"""
    qual_sents = []
    flags = slot_gate_filter.query_filter(dialogue_idx, turn_id, sentences, turn_label, thresh)
    for idx, flag in enumerate(flags):
        if flag:
            qual_sents.append(sentences[idx])

    return qual_sents


def subsitute(new_turn, ori_turn):
    """substitute value appear in the turn label
    """
    new_utter = (" " + ori_turn["transcript"] + " ")
    new_slot_dict = {}
    for domain_slot, value in new_turn["turn_label"]:
        new_slot_dict[domain_slot] = value

    for domain_slot, value in ori_turn["turn_label"]:

        search_span = re.search(r'[?,.! ]' + value + r'[?,.! ]', new_utter)
        if search_span:
            new_value = new_slot_dict[domain_slot]
            new_utter = new_utter[:search_span.start() + 1] + new_value + new_utter[search_span.end() - 1:]

    new_utter = new_utter.strip()
    if new_utter == ori_turn["transcript"]:
        return ""
    else:
        return new_utter

def decide_check_slot_gate(user_act, slot_gate_filter):
    labels = slot_gate_filter.processor.labels
    for domain, slots in user_act["Inform"].items():
        if domain in labels:
            for slot in slots:
                if slot[0] in labels[domain]:
                    return True
    return False


def filter_out_of_domain(domains):
    for domain in domains:
        if domain not in EXPERIMENT_DOMAINS:
            return True

    return False


def main(seed, size=1, ref_rate=0.6, out_path=None):
    """-----------------------------------argument setting begins-----------------------------------------------"""
    args_seed = int(seed)
    args_size = int(size)
    args_model_type = "t5"
    args_eval_data_file = "../MultiWOZ-coref/MultiWOZ2_3/train.json"
    args_model_name_or_path = "./cuda_model/checkpoint-36000"
    args_gene_data_save_dir = "./cuda_data"
    args_length = 512
    args_stop_token = None
    args_temperature = 1.0
    args_repetition_penalty = 1.0
    args_k = 0
    args_num_beams = 5
    args_p = 1.0
    args_no_cuda = False
    args_thresh = 0.5
    args_do_sample = False
    args_num_return_sequences = args_num_beams
    args_shuffle_turn_label = True
    """-----------------------------------coco control settings ----------------------------------------------"""
    args_method = ["cuda"]
    args_slot_value_dict = "out_cuda"
    args_classifier_filter = True
    args_slot_gate_filter = True
    args_immutable_inform = True
    args_max_inform = 3
    args_max_refer = 2
    args_try_refer_rate = float(ref_rate)
    args_reply_req_rate = 0.9
    args_confirm_rate = 0.7
    args_new_domain_rate = 0.8
    """-----------------------------------argument setting ends----------------------------------------------"""
    if args_classifier_filter:
        print("Add classifier filter")
        from classifier_filter.bert_filter import BERTFilter
        classifier_filter = BERTFilter(args_eval_data_file)
        print(len(classifier_filter.processor))
    
    if args_slot_gate_filter:
        print("Add slot_gate filter")
        from slot_gate_classifier_filter.slot_gate_filter import BERTSlotGateFilter
        slot_gate_filter = BERTSlotGateFilter(args_eval_data_file)
        print(len(slot_gate_filter.processor))

    assert args_model_type == "t5"
    args_device = torch.device("cuda" if torch.cuda.is_available() and not args_no_cuda else "cpu")
    args_n_gpu = 0 if args_no_cuda else torch.cuda.device_count()
    set_seed(args_seed, args_n_gpu)
    try:
        args_model_type = args_model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args_model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args_model_name_or_path)
    model = model_class.from_pretrained(args_model_name_or_path)
    model.to(args_device)
    args_length = adjust_length_to_model(args_length, max_sequence_length=model.config.n_positions)

    dataset = MultiWOZForT5_Interact(data_dir=args_eval_data_file, tokenizer=tokenizer,
                                     shuffle_turn_label=args_shuffle_turn_label)
    data_domain_maps = build_dict(args_eval_data_file)
    success_cuda = 0
    success_refer = 0
    save_info = {}
    # debug_info = {}
    print(len(dataset), " data points in total")
    for size in range(args_size):
        for idx in tqdm(range(len(dataset))):

            dialogue_idx, turn_idx = dataset.get_dialID_turnID(idx)
            if filter_out_of_domain(data_domain_maps[dialogue_idx]):
                continue
            # print(dialogue_idx)
            ori_turn = dataset.data[idx]
            new_turn = copy.deepcopy(ori_turn)
            success = False
            refer_success = False
            # debug_info[str(dialogue_idx) + str(turn_idx) + str(size)] = {}
            if "cuda" in args_method:
                new_turn, refer_success = counterfactual_goal_generator(new_turn, 
                                                        args_try_refer_rate, 
                                                        args_max_refer,
                                                        args_reply_req_rate, 
                                                        args_confirm_rate, 
                                                        args_new_domain_rate,
                                                        slot_value_dict=SLOT_VALUE_DICT[args_slot_value_dict],
                                                        immutable_inform=args_immutable_inform
                                                        )
                dataset.prompt_dict = new_turn["user_act"]
                encoded_prompt = dataset[idx]
                encoded_prompt = torch.tensor(encoded_prompt, dtype=torch.long).view(1, -1)
                encoded_prompt = encoded_prompt.to(args_device)
                if encoded_prompt.size()[-1] == 0:
                    input_ids = None
                else:
                    input_ids = encoded_prompt

                output_sequences = model.generate(
                    input_ids=input_ids,
                    max_length=args_length + len(encoded_prompt[0]),
                    temperature=args_temperature,
                    top_k=args_k,
                    top_p=args_p,
                    num_beams=args_num_beams,
                    repetition_penalty=args_repetition_penalty,
                    do_sample=args_do_sample,
                    num_return_sequences=args_num_return_sequences,
                )
                # Remove the batch dimension when returning multiple sequences
                if len(output_sequences.shape) > 2:
                    output_sequences.squeeze_()

                generated_sequences = []
                for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                    generated_sequence = generated_sequence.tolist()
                    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                    text = text[:text.find(args_stop_token) if args_stop_token else None]
                    generated_sequences.append(text)
                
                slot_gate_filter_sequences = None
                check_slot_gate = False
                if args_classifier_filter:
                    classifier_filter_sequences = classifier_filtering(classifier_filter, dialogue_idx, turn_idx,
                                                                generated_sequences, new_turn["turn_label"],
                                                                args_thresh)
                    if args_slot_gate_filter and len(classifier_filter_sequences) > 0:
                        check_slot_gate = decide_check_slot_gate(new_turn["user_act"], slot_gate_filter)
                        if check_slot_gate:
                            slot_gate_filter_sequences = slot_gate_filtering(slot_gate_filter, dialogue_idx, turn_idx,
                                                                classifier_filter_sequences, new_turn["user_act"], args_thresh)
                            best_seq = match_filtering(new_turn, ori_turn, slot_gate_filter_sequences)
                        else:
                            best_seq = match_filtering(new_turn, ori_turn, classifier_filter_sequences)
                    else:
                        best_seq = match_filtering(new_turn, ori_turn, classifier_filter_sequences)
                else:
                    best_seq = match_filtering(new_turn, ori_turn, generated_sequences)

                if best_seq:
                    new_turn["transcript"] = best_seq
                    new_turn = modify_belief_state_domain_slot(new_turn)
                    new_turn = update_belief_state_with_new_turn_label(new_turn)
                    success = True
                # else:
                #     debug_info[str(dialogue_idx) + str(turn_idx) + str(size)] = new_turn
                #     debug_info[str(dialogue_idx) + str(turn_idx) + str(size)]["generated_sequences"] = generated_sequences
                #     debug_info[str(dialogue_idx) + str(turn_idx) + str(size)]["pass_classifier"] = classifier_filter_sequences
                #     if check_slot_gate:
                #         debug_info[str(dialogue_idx) + str(turn_idx) + str(size)]["pass_slot_gate"] = slot_gate_filter_sequences
                # new_turn = copy.deepcopy(ori_turn)

            if success:
                success_cuda += 1
                if refer_success == 1 or refer_success == 2:
                    success_refer += 1

            else:
                new_turn = copy.deepcopy(ori_turn)
                new_turn = modify_belief_state_domain_slot(new_turn)
                new_turn = generate_origin_turn_label(new_turn)
                new_turn = update_belief_state(new_turn)
                new_turn["transcript"] = new_turn["text"]
                refer_success = False
            save_info[str(dialogue_idx) + str(turn_idx) + str(size)] = {"success": success,
                                                            "coreference": refer_success,
                                                            "context": new_turn["context"],
                                                            "new_utter": new_turn["transcript"],
                                                            "new_turn_label": new_turn["new_turn_label"],
                                                            "belief_state": new_turn["belief_state"]}

    save_info["success cuda rate"] = success_cuda / (idx + 1)
    print("success cuda generation rate: ", success_cuda / (idx + 1))
    save_info["success refer rate"] = success_refer / (idx + 1)
    print("success refer rate: ", success_refer / (idx + 1))

    if out_path is None:
        args_special_str = "-".join(args_method) + "_" + args_slot_value_dict

        if args_classifier_filter:
            args_special_str += "_classifier"

        if args_slot_gate_filter:
            args_special_str += "_slot-gate"    

        if args_max_inform:
            args_special_str += ("_max-" + str(args_max_inform))

        if args_reply_req_rate:
            args_special_str += ("_replyReqRate-" + str(args_reply_req_rate))

        if args_confirm_rate:
            args_special_str += ("_confirmRate-" + str(args_confirm_rate))

        if args_new_domain_rate:
            args_special_str += ("_newDomainRate-" + str(args_new_domain_rate))
        
        args_special_str += ("_tryReferRate-" + str(args_try_refer_rate))

        saved_file_name = args_special_str + "_seed_" + str(args_seed) + ".json"
        if not os.path.exists(args_gene_data_save_dir):
            os.makedirs(args_gene_data_save_dir)

        with open(os.path.join(args_gene_data_save_dir, saved_file_name), "w") as f:
            json.dump(save_info, f, indent=4)
    else:
        args_special_str = out_path
        with open(args_special_str, "w") as f:
            json.dump(save_info, f, indent=4)


    # with open(os.path.join(args_gene_data_save_dir, "[DEBUG]" + saved_file_name), "w") as f:
    #     json.dump(debug_info, f, indent=4)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], size=sys.argv[2], ref_rate=float(sys.argv[3]))
    else:
        main(sys.argv[1], size=sys.argv[2], ref_rate=float(sys.argv[3]), out_path=sys.argv[4])
