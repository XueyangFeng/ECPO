export CUDA_VISIBLE_DEVICES=0
# Define hyperparameters
LEARNING_RATE=1.0e-7
NUM_EPOCHS=3.0

# Define the output directory dynamically based on learning rate and num_epochs
OUTPUT_DIR="/data/fengxueyang/ecpo/framework_fine-tuning/rag/ecpo_v1/Book/lr${LEARNING_RATE}_epochs${NUM_EPOCHS}"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the training command directly with command-line arguments, without using YAML
nohup llamafactory-cli train \
  --model_name_or_path /data/fengxueyang/.cache/modelscope/hub/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --adapter_name_or_path /data/fengxueyang/ecpo/framework_fine-tuning/rag/sft/Book/lr5.0e-5_epochs3.0\
  --stage dpo \
  --do_train true \
  --finetuning_type lora \
  --lora_target all \
  --dataset amazon_book_rag_ecpo_v1 \
  --template llama3 \
  --cutoff_len 2048 \
  --overwrite_cache true \
  --preprocessing_num_workers 16 \
  --output_dir $OUTPUT_DIR \
  --logging_steps 10 \
  --save_steps 500 \
  --plot_loss true \
  --overwrite_output_dir true \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_EPOCHS \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 true \
  --ddp_timeout 180000000 \
  --val_size 0.1 \
  --per_device_eval_batch_size 1 \
  --eval_strategy steps \
  --eval_steps 500 \
  > $OUTPUT_DIR/train.log 2>&1 &
