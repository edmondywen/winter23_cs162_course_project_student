TASK_NAME="com2sense"
DATA_DIR="datasets/com2sense"
MODEL_TYPE="distilroberta-base"

python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_not_load_optimizer \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 100.0 \
  --max_seq_length 128 \
  --output_dir "${TASK_NAME}/ckpts" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 200 \
  --logging_steps 5 \
  --warmup_steps 100 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --iters_to_eval 4000 5000 \
  --overwrite_output_dir \
  # --max_eval_steps 1000 \