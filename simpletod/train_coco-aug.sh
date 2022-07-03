export CUDA_VISIBLE_DEVICES=0
MODEL="gpt2"
MODEL_NAME="gpt2"
BATCH=2
OUTPUT_DIR="cuda_more_new"
TRAIN_FILE=./resources/train.cuda_new_aug_history_belief
# OUTPUT_DIR="cuda_more_new_withoutRef"
# TRAIN_FILE=./resources/train.cuda_new_withoutRef_aug_history_belief
TEST_FILE=./resources/val.history_belief

mkdir -p $OUTPUT_DIR


python main.py \
--output_dir=$OUTPUT_DIR \
--model_type=$MODEL \
--model_name_or_path=$MODEL_NAME \
--do_train \
--train_data_file=$TRAIN_FILE \
--do_eval \
--eval_data_file=$TEST_FILE \
--evaluate_during_training \
--logging_steps 5000 \
--per_gpu_train_batch_size $BATCH \
--save_steps 5000 \
--num_train_epochs 30
# --num_train_epochs 6
# --save_steps 10000 \
