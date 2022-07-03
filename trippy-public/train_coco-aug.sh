export CUDA_VISIBLE_DEVICES=0
TASK="multiwoz21"
DATA_DIR="data/MULTIWOZ2.1"


# Project paths etc. ----------------------------------------------
aug_file="../cuda/cuda_data/cuda_out_cuda_classifier_slot-gate_max-3_replyReqRate-0.9_confirmRate-0.7_newDomainRate-0.8_tryReferRate-0.6_seed_0.json"
OUT_DIR=cuda
# config="./cuda_new_withoutRef/checkpoint-70810"
if [ ! -d "${OUT_DIR}" ]; then
  mkdir -p ${OUT_DIR}
fi
# Main ------------------------------------------------------------

for step in train dev test; do
    args_add=""
    if [ "$step" = "train" ]; then
	args_add="--do_train --predict_type=dummy"
    elif [ "$step" = "dev" ] || [ "$step" = "test" ]; then
	args_add="--do_eval --predict_type=${step}"
    fi

    python3 run_dst.py \
	    --task_name=${TASK} \
	    --data_dir=${DATA_DIR} \
	    --dataset_config=dataset_config/${TASK}.json \
	    --model_type="bert" \
	    --model_name_or_path="bert-base-uncased" \
	    --do_lower_case \
	    --aug_file=${aug_file} \
	    --learning_rate=1e-4 \
	    --num_train_epochs=10 \
	    --max_seq_length=180 \
	    --per_gpu_train_batch_size=16 \
	    --per_gpu_eval_batch_size=1 \
	    --output_dir=${OUT_DIR} \
	    --save_epochs=2 \
	    --logging_steps=10 \
	    --warmup_proportion=0.1 \
	    --eval_all_checkpoints \
	    --adam_epsilon=1e-6 \
	    --label_value_repetitions \
	    --swap_utterances \
	    --append_history \
	    --use_history_labels \
	    --delexicalize_sys_utts \
	    --class_aux_feats_inform \
	    --class_aux_feats_ds \
	    ${args_add} \
	    2>&1 | tee ${OUT_DIR}/${step}.log
		# --config_name=${config} \

    if [ "$step" = "dev" ] || [ "$step" = "test" ]; then
    	python3 metric_bert_dst.py \
    		${TASK} \
			dataset_config/${TASK}.json \
    		"${OUT_DIR}/pred_res.${step}*json" \
    		2>&1 | tee ${OUT_DIR}/eval_pred_${step}.log
    fi
done
