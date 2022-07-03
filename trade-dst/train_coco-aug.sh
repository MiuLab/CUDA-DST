export CUDA_VISIBLE_DEVICES=0
aug_file="../cuda/cuda_new_data/cuda_out_cuda_classifier_binary_max-3_replyReqRate-0.9_confirmRate-0.7_newDomainRate-0.8_tryReferRate-0.0_bool_confirm_single_recommend_dontcare_seed_0.json"
# aug_file="../coco-dst/coco_data/coco-vs_rare_out_domain_train_classifier_change_add-2-max-3_drop-1_seed_0.json"
python3 myTrain.py -dec=TRADE -bsz=16 -dr=0.2 -lr=0.001 -le=1 -patience=2 -out_dir="cuda_new_withoutRef" -aug_file=${aug_file}
