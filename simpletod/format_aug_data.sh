ori_dial_file="../multiwoz/MultiWOZ_2.1/train_dials.json"
# aug_dials_file="../coco-dst_refer/coco_data/coco-vs_rare_out_domain_train_classifier_change_add-2-max-3_drop-1_refer-1_inform_seed_0.json"
aug_dials_file="../cuda/cuda_new_data/cuda_out_cuda_classifier_binary_max-3_replyReqRate-0.9_confirmRate-0.7_newDomainRate-0.8_tryReferRate-0.0_bool_confirm_single_recommend_dontcare_seed_0.json"
save_name="cuda_new_withoutRef"
python format_aug_data.py --aug_dials_file=${aug_dials_file} --ori_dial_file=${ori_dial_file} --save_name=${save_name}