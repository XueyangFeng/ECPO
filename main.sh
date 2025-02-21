#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# 定义参数变量
MODE="test"
DOMAIN="Book"
CRS_TYPE="ActCRS" #Choose from [ReActCRS, ActCRS, RAGCRS, ZSCRS, MACRS]
PERSONA_PATH="user_simulator/persona/${DOMAIN}_${MODE}.jsonl"
CONFIG_PATH="config/api_config.json"
FORMAT_PATH="config"
USER_MODEL="openai_mini"
CRS_MODEL="llama"
INDEX_FILE="raw_data/emb/${DOMAIN}/faiss_index.bin"
METADATA_FILE="raw_data/emb/${DOMAIN}/metadata.json"
EMB_MODEL_PATH="crs/tools/all-MiniLM-L6-v2"
NUM_USERS=100
NUM_THREADS=20
CONVERSATION_ROUNDS=5
CRS_TEMPERATURE=0.0

OUTPUT_DIR="${DOMAIN}_result/${CRS_TYPE}"

# 输出日志文件名，包含时间戳，并根据模型名称组织文件夹
LOG_FILE="${OUTPUT_DIR}/${CRS_MODEL}/main_${CRS_TYPE}_$(date +'%Y%m%d_%H%M%S').log"

mkdir -p "$OUTPUT_DIR/${CRS_MODEL}"


nohup python main.py \
  --domain "$DOMAIN" \
  --mode "$MODE" \
  --crs_type "$CRS_TYPE" \
  --persona_path "$PERSONA_PATH" \
  --config_path "$CONFIG_PATH" \
  --format_path "$FORMAT_PATH" \
  --user_model "$USER_MODEL" \
  --crs_model "$CRS_MODEL" \
  --index_file "$INDEX_FILE" \
  --metadata_file "$METADATA_FILE" \
  --emb_model_path "$EMB_MODEL_PATH" \
  --num_users "$NUM_USERS" \
  --num_threads "$NUM_THREADS" \
  --conversation_rounds "$CONVERSATION_ROUNDS" \
  --output_dir "$OUTPUT_DIR" \
  --crs_temperature "$CRS_TEMPERATURE" \
  > "$LOG_FILE" 2>&1 &

echo "Script is running in the background. Output is being logged to $LOG_FILE"