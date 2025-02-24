export CUDA_VISIBLE_DEVICES=0
# Define hyperparameters
LEARNING_RATE=5.0e-5
NUM_EPOCHS=3.0

# Define the output directory dynamically based on learning rate and num_epochs
OUTPUT_DIR="/your/lora/path/sft/Game/lr${LEARNING_RATE}_epochs${NUM_EPOCHS}"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the training command directly with command-line arguments, without using YAML
nohup llamafactory-cli train \
  --model_name_or_path /your/base/model/path \
  --stage sft \
  --do_train true \
  --finetuning_type lora \
  --lora_target all \
  --dataset amazon_game_sft \
  --template llama3 \
  --cutoff_len 2048 \
  --overwrite_cache true \
  --preprocessing_num_workers 16 \
  --output_dir $OUTPUT_DIR \
  --logging_steps 10 \
  --save_steps 500 \
  --plot_loss true \
  --overwrite_output_dir true \
  --per_device_train_batch_size 1 \
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
