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
import json

def set_seed(args_seed, args_n_gpu):
    np.random.seed(args_seed)
    torch.manual_seed(args_seed)
    random.seed(args_seed)
    if args_n_gpu > 0:
        torch.cuda.manual_seed_all(args_seed)


def main():
    """-----------------------------------argument setting begins-----------------------------------------------"""
    args_seed = 0
    args_eval_data_file = "./eval_file/coco-vs_rare_out_domain_test_classifier_change_add-2-max-3_drop-1_seed_1.json"
    args_seed = args_seed
    args_no_cuda = False
    args_ignore_none_and_dontcare = False  # SimpleTOD uses this setting for evaluation.
    args_target_model = "trippy"  # Possible models: ["trade","trippy","simpletod"]
    # args_eval_result_save_dir = "../all_without_no_dontcare/size1_seed0"
    args_eval_result_save_dir = "./eval_result"
    args_eval_result_file_name = "cuda_eval.json"
    """-----------------------------------argument setting ends----------------------------------------------"""

    if ("trade" in args_target_model):
        print("Evaluating Trade")
        sys.path.append("../trade-dst")
        from interface4eval_trade import Trade_DST
        target_model = Trade_DST()
    elif ("trippy" in args_target_model):
        print("Evaluating Trippy")
        sys.path.append("../trippy-public")
        from interface4eval_trippy import TripPy_DST
        target_model = TripPy_DST()
    elif ("simpletod" in args_target_model):
        print("Evaluating simpletod")
        sys.path.append("../simpletod")
        from interface4eval_simpletod import SimpleToD_DST
        target_model = SimpleToD_DST()

    args_device = torch.device("cuda" if torch.cuda.is_available() and not args_no_cuda else "cpu")
    args_n_gpu = 0 if args_no_cuda else torch.cuda.device_count()
    set_seed(args_seed, args_n_gpu)
    
    success_gen = 0
    new_correct_pred = 0
    ori_correct_pred = 0
    save_info = {}
    origin = None
    new = None
    all_dialogue_id = {}
    
    with open(args_eval_data_file, "r") as f:
        all_data = json.load(f)
        origin = all_data["origin"]
        new = all_data["new"]
        success_gen = all_data["success rate"]
        for id in origin.keys():
            try:
                int(id)
            except:
                all_dialogue_id[id] = len(origin[id]["dialogue"])

    idx = 0
    for dialogue_idx, turn_num in tqdm(all_dialogue_id.items()):
        turn_num = all_dialogue_id[dialogue_idx]
        for turn_idx in range(turn_num):
            new_turn = new[dialogue_idx]["dialogue"][turn_idx]
            if ("trade" in args_target_model):
                new_prediction = target_model.dst_query(new[dialogue_idx], turn_idx, new_turn["transcript"],
                                                        args_ignore_none_and_dontcare)
            elif ("trippy" in args_target_model):
                new_prediction = target_model.dst_query(dialogue_idx, turn_idx, new_turn["transcript"],
                                                        new_turn["turn_label"], args_ignore_none_and_dontcare)
            elif ("simpletod" in args_target_model):
                new_prediction = target_model.dst_query(new[dialogue_idx], turn_idx, new_turn["transcript"],
                                                        args_ignore_none_and_dontcare)

            ori_turn = origin[dialogue_idx]["dialogue"][turn_idx]
            if ("trade" in args_target_model):
                ori_prediction = target_model.dst_query(origin[dialogue_idx], turn_idx, ori_turn["transcript"],
                                                        args_ignore_none_and_dontcare)
            elif ("trippy" in args_target_model):
                ori_prediction = target_model.dst_query(dialogue_idx, turn_idx, ori_turn["transcript"],
                                                        ori_turn["turn_label"], args_ignore_none_and_dontcare)
            elif ("simpletod" in args_target_model):
                ori_prediction = target_model.dst_query(origin[dialogue_idx], turn_idx, ori_turn["transcript"],
                                                        args_ignore_none_and_dontcare)

        
            new_correct_pred += new_prediction['Joint Acc']
            ori_correct_pred += ori_prediction['Joint Acc']
            save_info[str(dialogue_idx) + str(turn_idx)] = {"ori_prediction": ori_prediction,
                                                            "new_prediction": new_prediction,
                                                            "new_utter": new_turn["transcript"],
                                                            "new_turn_label": new_turn["turn_label"],
                                                            "ori_utter": ori_turn["transcript"],
                                                            "ori_turn_label": ori_turn["turn_label"]}

            if (idx % 100 == 0):
                print("success generation rate: ", success_gen)
                print("avg new joint acc: ", new_correct_pred / (idx + 1))
                print("avg original joint acc: ", ori_correct_pred / (idx + 1))
            idx += 1

    save_info["avg new joint acc"] = new_correct_pred / (idx + 1)
    save_info["avg ori joint acc"] = ori_correct_pred / (idx + 1)
    save_info["success rate"] = success_gen

    save_path = os.path.join(args_eval_result_save_dir, args_target_model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    saved_file_name = args_eval_result_file_name
    with open(os.path.join(save_path, saved_file_name), "w") as f:
        json.dump(save_info, f, indent=4)


if __name__ == "__main__":
    main()
