import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import sys
import os
from refine_prompts import expression_refiner_template
# 获取当前工作目录
current_dir = os.getcwd()

# 获取父级目录路径
parent_dir = os.path.abspath(os.path.join(current_dir, '../../../../'))

# 将父级目录添加到 sys.path
sys.path.insert(0, parent_dir)
from collections import defaultdict
from datetime import datetime  # Import datetime for timestamp functionality
from model.model import OpenAIClient

# 从 JSONL 文件加载数据（逐行加载）
def load_jsonl(file_path: str, num_lines=None):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if num_lines is not None and i >= num_lines:
                break
            data.append(json.loads(line))
    return data

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def parse_json(response):
    if response.startswith("```json"):
        response = response[len("```json"):].strip()
    if response.endswith("```"):
        response = response[:-len("```")].strip()
    return json.loads(response)


def refine_single_trajectory(openai_client, input, original_response, generative_reward):
    def _build_expression_rewriter_prompt():
        prompt = expression_refiner_template.format(
            Scratchpad=input,
            Original_response=original_response,
            Generative_Reward=generative_reward
        )
        return prompt
    """
    使用 openai_client 对单个对话进行refine。

    Parameters:
    - openai_client: The OpenAI API client.
    - original_text (str): 原始对话字符串
    - refine_prompt (str): refine的prompt模板
    - refine feedback (str):

    Returns:
    - str: refine 后的文本
    """
    prompt =_build_expression_rewriter_prompt()
    try:
        response = openai_client.get_single_chat_completion(prompt).strip()
        return response
    except Exception as e:
        print(f"Refine Error: {e}")
        raise


def is_valid_format(text):
    """检测字符串是否严格符合 Ask[Question]、Recommend[Answer]、Response[Content]、Search[Keyword] 格式"""
    if not text:
        return False
    # 正则匹配四种格式之一
    pattern = r"^(Ask\[[^\[\]]+\]|Recommend\[[^\[\]]+\]|Response\[[^\[\]]+\]|Search\[[^\[\]]+\])$"
    return re.match(pattern, text) is not None


def get_strategy(text):
    """提取策略部分（Ask、Recommend、Response、Search）"""
    if not is_valid_format(text):
        return None
    # 提取策略的名字
    match = re.match(r"^(Ask|Recommend|Response|Search)\[.*\]$", text)
    return match.group(1) if match else None


def refine_trajectories(tasks, openai_client, max_workers=5):
    """
    并发对多个对话进行 refine 处理。
    """


    sft_results = []
    dpo_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_user = {
            executor.submit(refine_single_trajectory, openai_client, t[1], t[2], t[4]): t  # 不再只存 t[0]
            for t in tasks if t[3] is not None and t[3] < 5
        }

        for future in tqdm(as_completed(future_to_user), total=len(future_to_user), desc="Refining Dialogues"):
            task = future_to_user[future]  # 这里的 task 是完整的 t，而不是 t[0]
            input = task[1]  # 直接从 task 中取 input
            original_output = task[2]
            generative_reward = task[4]
            refinement = future.result()
            try:
                refinement = parse_json(refinement)["refinement"]
                if is_valid_format(refinement):
                    if refinement != original_output:            
                        sft_results.append({
                            "system": "You are a helpful assistant",
                            "instruction": input,
                            "input": "",
                            "output": refinement,
                            "generative_reward": generative_reward
                        })
                        dpo_results.append({
                            "system": "You are a helpful assistant",
                            "instruction": input,
                            "input": "",
                            "chosen": refinement,
                            "rejected": original_output,
                            "generative_reward": generative_reward
                        })
            except:
                continue

    return sft_results, dpo_results

def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def main():
    # 读入一个文件的路径
    input_file = "./amazon_game_ecpo_raw.json"
    
    raw_data = load_json(input_file)
    tasks = []
    for i in range(len(raw_data)):
        if raw_data[i]["exp_reward"]!= None:
            tasks.append((i, raw_data[i]["Instruction"], raw_data[i]["output"], raw_data[i]["exp_reward"], raw_data[i]["exp_reason"]))
    print(tasks[0])
    print(len(tasks[0]))
    #for i in range(5):

    # 生成时间戳文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join("./", timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config_path = "../../../../config/api_config.json"
    config = load_config(config_path)["openai"]
    openai_client = OpenAIClient(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_path=config["model_path"],
    )

    mini_config = load_config(config_path)["openai_mini"]
    openai_mini_client = OpenAIClient(
        base_url=mini_config["base_url"],
        api_key=mini_config["api_key"],
        model_path=mini_config["model_path"],    
    )

    sft_results, dpo_results = refine_trajectories(tasks, openai_mini_client, max_workers=10)

    # 可选择保存到一个新的 JSON 文件
    sft_output_file = './amazon_game_ecpo_sft.json'  # 请替换为你想保存的路径
    dpo_output_file = './amazon_game_ecpo_dpo.json'  # 

    with open(sft_output_file, 'w', encoding='utf-8') as outfile:
        json.dump(sft_results, outfile, ensure_ascii=False, indent=4)

    with open(dpo_output_file, 'w', encoding='utf-8') as outfile:
        json.dump(dpo_results, outfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
