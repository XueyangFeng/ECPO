import sys
sys.path.append('/data/fxy/ecpo')
from model.model import OpenAIClient
import json
from tqdm import tqdm

# 从配置文件中加载参数
def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def summarize_text(openai_client, descriptions):
    # 将描述合并为适合批量处理的 prompt
    prompts = [f"Describe this store in natural language:\n\n{desc}" for desc in descriptions]
    # 获取模型的批量总结响应
    summaries = openai_client.get_multi_chat_completions(prompts)
    return summaries  # 提取每条摘要内容

def process_descriptions(input_file, output_file, openai_client, batch_size=5):
    # 打开输入和输出文件
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        batch_users = []  # 存储用户数据的批次
        descriptions = []  # 存储每批次的所有描述

        # 使用 tqdm 包装文件行，以显示进度条
        for line in tqdm(infile, desc="Processing users", unit="user"):
            user_data = json.loads(line.strip())  # 每行对应一个用户
            user_descriptions = []  # 当前用户的所有描述

            # 为每个用户收集所有 Review 的描述
            for review in user_data["ReviewList"]:
                description = {
                    "BusinessID": review.get("BusinessID"),
                    "BusinessName": review.get("BusinessName"),
                    "Address": review.get("Address"),
                    "City": review.get("City"),
                    "State": review.get("State"),
                    "PostalCode": review.get("PostalCode"),
                    "Categories": review.get("Categories"),
                    "Attributes": review.get("Attributes"),
                    "Hours": review.get("Hours")
                }
                user_descriptions.append(description)
                descriptions.append(description)  # 将描述添加到总批次描述中

            # 保存用户数据和对应描述，以便替换 Summary
            batch_users.append((user_data, user_descriptions))

            # 达到批量大小时，处理并替换批次描述
            if len(batch_users) == batch_size:
                # 调用 LLM API 进行总结
                summaries = summarize_text(openai_client, descriptions)

                # 将 summary 替换回每个用户的对应 review
                summary_index = 0
                for user_data, user_descriptions in batch_users:
                    for review in user_data["ReviewList"]:
                        review["Description"] = summaries[summary_index]
                        summary_index += 1
                        # 删除所有商家信息，只保留 Description
                        review.pop("BusinessID", None)
                        review.pop("Address", None)
                        review.pop("City", None)
                        review.pop("State", None)
                        review.pop("PostalCode", None)
                        review.pop("Attributes", None)
                        review.pop("Hours", None)

                    # 将处理后的用户数据写入文件
                    outfile.write(json.dumps(user_data, ensure_ascii=False) + '\n')

                # 清空批次数据
                batch_users = []
                descriptions = []

        # 处理文件末尾不足批量大小的剩余数据
        if batch_users:
            summaries = summarize_text(openai_client, descriptions)
            summary_index = 0
            for user_data, user_descriptions in batch_users:
                for review in user_data["ReviewList"]:
                    review["Description"] = summaries[summary_index]
                    summary_index += 1
                outfile.write(json.dumps(user_data, ensure_ascii=False) + '\n')

    print(f"Processed data has been saved to {output_file} with summaries only.")

# 示例：外部文件中调用
if __name__ == "__main__":
    # 加载配置
    config = load_config("../../config/api_config.json")["openai_mini"]

    # 初始化 OpenAI 客户端
    openai_client = OpenAIClient(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_path=config["model_path"]
    )

    # Example usage
    input_file = 'processed_data.jsonl'  # JSONL input file path
    output_file = 'processed_data_with_summaries.jsonl'  # Output JSONL file path with summaries only

    process_descriptions(input_file, output_file, openai_client, batch_size=16)
