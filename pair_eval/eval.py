# Install required packages (uncomment if not already installed)
# !pip install openai tqdm

import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import sys
import os
from collections import defaultdict
from datetime import datetime  # Import datetime for timestamp functionality

# 获取当前工作目录
current_dir = os.getcwd()

# 获取父级目录路径
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))

# 将父级目录添加到 sys.path
sys.path.insert(0, parent_dir)

from model.model import OpenAIClient


def parse_log_file(filepath):
    """
    解析 .log 文件，返回一个字典。
    key 是用户 ID（0-99），value 是对话记录的字符串列表。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        log_data = f.read()

    # 使用正则表达式分割记录，匹配由多行等号组成的分隔符
    # 例如：=====================================================================================================================
    records = re.split(r'\n=+\n', log_data)

    user_data = defaultdict(list)

    for idx, record in enumerate(records):
        record = record.strip()
        if not record:
            continue  # 跳过空记录

        # 解析头部的 User ID 和 Item ID
        lines = record.split('\n')
        if not lines:
            print(f"记录 {idx} 是空的，跳过。")
            continue

        header_line = lines[0].strip()
        # 正则表达式匹配 "User <id>  Item <id>:"
        match = re.match(r'^User\s+(\d+)\s+Item\s+\d+:', header_line)
        if not match:
            print(f"无法匹配头部信息 (记录 {idx}) : {header_line}")
            continue  # 跳过无法匹配的记录

        user_id = int(match.group(1))

        # 提取对话部分：只保留以 'user:' 或 'assistant:' 开头的行
        dialogue_lines = []
        for line in lines[1:]:
            line = line.strip()
            if line.startswith('user:') or line.startswith('assistant:'):
                dialogue_lines.append(line)
            else:
                # 遇到非对话行时，停止提取对话
                break

        # 将对话行合并为一个字符串
        raw_text = '\n'.join(dialogue_lines)

        # 添加到字典中
        user_data[user_id].append(raw_text)

    return dict(user_data)


def parse_jsonl_file(filepath):
    """
    解析 .jsonl 文件，返回一个字典。
    假设每行 JSON 包含 'user_id', 'dialogue', 'refined_dialogue' 字段。
    """
    user_data = defaultdict(list)

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                entry = json.loads(line)
                user_id = entry.get('user_id') or entry.get('dialogue_id')  # 根据实际字段调整
                refined_dialogue = entry.get('refined_dialogue')

                if refined_dialogue is not None:
                    user_data[user_id].append(refined_dialogue)
            except json.JSONDecodeError as jde:
                print(f"JSON解析错误 (第{line_num}行): {jde}")
                continue  # 跳过解析错误的行

    return dict(user_data)


def parse_trajectory_file(filepath):
    """
    根据文件扩展名调用相应的解析函数，返回一个字典。
    """
    if filepath.endswith('.log'):
        return parse_log_file(filepath)
    elif filepath.endswith('.jsonl'):
        return parse_jsonl_file(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def refine_single_trajectory(openai_client, original_text, refine_prompt):
    """
    使用 openai_client 对单个对话进行 refine。

    Parameters:
    - openai_client: The OpenAI API client.
    - original_text (str): 原始对话字符串
    - refine_prompt (str): refine的prompt模板

    Returns:
    - str: refine 后的文本
    """
    prompt = refine_prompt.format(traj=original_text)
    try:
        response = openai_client.get_single_chat_completion(prompt).strip()
        return response
    except Exception as e:
        print(f"Refine Error: {e}")
        return original_text  # 出错则返回原始文本


def refine_trajectories(parsed_data, refine_prompt, openai_client, max_workers=5):
    """
    并发对多个对话进行 refine 处理。

    Returns:
    - list: 列表，每个元素为 (user_id, original_text, refined_text)
    """
    tasks = []
    for user_id, dialogues in parsed_data.items():
        if len(dialogues) >= 2:
            original_text = dialogues[0]
            refined_text = dialogues[1]
            tasks.append((user_id, original_text, refined_text))
        elif len(dialogues) == 1:
            original_text = dialogues[0]
            tasks.append((user_id, original_text, None))  # 没有 refined_dialogue

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_user = {
            executor.submit(refine_single_trajectory, openai_client, t[1], refine_prompt): t[0]
            for t in tasks if t[2] is None  # 仅对需要 refine 的对话提交任务
        }

        for future in tqdm(as_completed(future_to_user), total=len(future_to_user), desc="Refining Dialogues"):
            user_id = future_to_user[future]
            original_text = parsed_data[user_id][0]

            refined_text = future.result()
            results.append((user_id, original_text, refined_text))
        
        # 如果 refined_dialogue 已存在，则不需要 refine
        for t in tasks:
            if t[2] is not None:
                results.append((t[0], t[1], t[2]))
    
    return results


def classify_single_pair(openai_client, user_id, traj_a, traj_b, eval_format):
    """
    Compares two dialogue trajectories and returns the evaluation result.
    
    Parameters:
    - openai_client: The OpenAI API client.
    - user_id (int): 用户 ID。
    - traj_a (str): Dialogue trajectory A。
    - traj_b (str): Dialogue trajectory B。
    - eval_format (str): The evaluation prompt template。
    
    Returns:
    - Dict: The evaluation result in JSON format。
    """
    prompt = eval_format.format(Traj_a=traj_a, Traj_b=traj_b)
    
    try:
        response = openai_client.get_single_chat_completion(prompt).strip()
        #print(f"User {user_id} API Response: {response}")  # 可选调试信息
        
        # 移除可能的 ```json 和 ``` 标记
        if response.startswith("```json"):
            response = response[len("```json"):].strip()
        if response.endswith("```"):
            response = response[:-len("```")].strip()
        
        # 提取第一个 '{' 到最后一个 '}' 之间的内容
        json_start = response.find('{')
        json_end = response.rfind('}')
        if json_start == -1 or json_end == -1:
            print(f"User {user_id} 无法找到有效的 JSON 内容。")
            return None
        json_str = response[json_start:json_end+1]
        
        # 解析 JSON
        result = json.loads(json_str)
        # 添加 user_id 到结果中
        result["User ID"] = user_id
        return result
    except json.JSONDecodeError as jde:
        print(f"User {user_id} JSON解析错误: {jde}")
        print(f"原始响应: {response}")
        return None
    except Exception as e:
        print(f"User {user_id} Error during OpenAI API call: {e}")
        return None


def classify_comments(comparison_pairs, eval_format, openai_client, max_workers=5, swap=False):
    """
    并发分类多个对话对。
    
    Parameters:
    - comparison_pairs (List[Tuple[int, str, str]]): 对话对列表，包含 user_id。
    - eval_format (str): 评估提示模板。
    - openai_client: The OpenAI API client。
    - max_workers (int): 使用的线程数。
    - swap (bool): 是否交换 Traj_a 和 Traj_b。
    
    Returns:
    - List[Dict]: 分类结果列表。
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {}
        for pair in comparison_pairs:
            user_id, traj_a, traj_b = pair
            if swap:
                future = executor.submit(classify_single_pair, openai_client, user_id, traj_b, traj_a, eval_format)
            else:
                future = executor.submit(classify_single_pair, openai_client, user_id, traj_a, traj_b, eval_format)
            future_to_pair[future] = pair
        
        for future in tqdm(as_completed(future_to_pair), total=len(comparison_pairs), desc="Classifying Dialogues"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                pair = future_to_pair[future]
                user_id = pair[0]
                print(f"User {user_id} Error during classification: {e}")
                results.append(None)  # 标记为未决定
    return results


def calculate_win_rate(results_a_vs_b, results_b_vs_a):
    """
    Calculates the average win rates for Traj A and Traj B based on the evaluation results.
    
    Parameters:
    - results_a_vs_b (List[Dict]): List of evaluation results for A vs B.
    - results_b_vs_a (List[Dict]): List of evaluation results for B vs A.
    
    Returns:
    - Dict: 包含各指标及最终分数的胜率和原始计数。
    """
    # 初始化统计计数器
    metrics = {
        "Flexibility": {"A": 0, "B": 0, "Tie": 0},
        "Coherence": {"A": 0, "B": 0, "Tie": 0},
        "User Guidance": {"A": 0, "B": 0, "Tie": 0},
        "Final Score": {"A": 0, "B": 0, "Tie": 0}
    }

    # 处理 A vs B 的结果
    for result in results_a_vs_b:
        if result is None:
            continue  # 忽略错误结果
        for metric in metrics.keys():
            if metric == "Final Score":
                score = result.get(metric, 0)
            else:
                score = result.get(metric, {}).get("Score", 0)
            if score == 1:
                metrics[metric]["A"] += 1
            elif score == -1:
                metrics[metric]["B"] += 1
            else:
                metrics[metric]["Tie"] += 1

    # 处理 B vs A 的结果（反转分数）
    for result in results_b_vs_a:
        if result is None:
            continue  # 忽略错误结果
        for metric in metrics.keys():
            if metric == "Final Score":
                score = result.get(metric, 0)
                score = -score  # 反转分数
            else:
                score = result.get(metric, {}).get("Score", 0)
                score = -score  # 反转分数
            if score == 1:
                metrics[metric]["A"] += 1
            elif score == -1:
                metrics[metric]["B"] += 1
            else:
                metrics[metric]["Tie"] += 1

    # 计算胜率，包括打平
    win_rates = {}
    for metric, counts in metrics.items():
        total = counts["A"] + counts["B"] + counts["Tie"]
        if total == 0:
            win_rate_a = 0
            win_rate_b = 0
        else:
            # 定义胜率为 (A wins + 0.5 * Ties) / (A wins + B wins + Ties)
            win_rate_a = (counts["A"] + 0.5 * counts["Tie"]) / total
            win_rate_b = (counts["B"] + 0.5 * counts["Tie"]) / total
        win_rates[f"{metric} Win Rate"] = {
            "A Wins": counts["A"],
            "B Wins": counts["B"],
            "Ties": counts["Tie"],
            "Win Rate A": win_rate_a,
            "Win Rate B": win_rate_b,
            "Total": total
        }
        # 打印每个指标的胜率
        print(f"{metric} Win Rate:")
        print(f"  Traj A wins: {counts['A']}")
        print(f"  Traj B wins: {counts['B']}")
        print(f"  Ties: {counts['Tie']}")
        print(f"  Traj A win rate: {win_rate_a:.2%}")
        print(f"  Traj B win rate: {win_rate_b:.2%}\n")

    return win_rates


def print_and_save_results(results_a_vs_b, results_b_vs_a, folder_name, label):
    """
    保存评估结果和计算胜率。
    
    Parameters:
    - results_a_vs_b (List[Dict]): A vs B 的评估结果。
    - results_b_vs_a (List[Dict]): B vs A 的评估结果。
    - folder_name (str): 保存结果的文件夹路径。
    - label (str): 评估标签，如 "Original vs Refined"。
    
    Returns:
    - Dict: 胜率计算结果。
    """
    # 创建文件夹
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    else:
        print(f"Folder already exists: {folder_name}")

    # 保存 A vs B 的评估结果
    raw_scores_a_vs_b = []
    for idx, result in enumerate(results_a_vs_b, start=1):
        if result is None:
            print(f"Skipping saving for evaluation {idx} (A vs B) due to None result.")
            continue
        user_id = result.get("User ID", idx)
        evaluation_filename = os.path.join(folder_name, f"evaluation_A_vs_B_user_{user_id}.json")
        with open(evaluation_filename, 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)
        raw_scores_a_vs_b.append(result)

    raw_scores_a_vs_b_filename = os.path.join(folder_name, "raw_scores_A_vs_B.json")
    with open(raw_scores_a_vs_b_filename, 'w', encoding='utf-8') as json_file:
        json.dump(raw_scores_a_vs_b, json_file, ensure_ascii=False, indent=4)
    print(f"All raw scores for A vs B have been saved to {raw_scores_a_vs_b_filename}")

    # 保存 B vs A 的评估结果
    raw_scores_b_vs_a = []
    for idx, result in enumerate(results_b_vs_a, start=1):
        if result is None:
            print(f"Skipping saving for evaluation {idx} (B vs A) due to None result.")
            continue
        user_id = result.get("User ID", idx)
        evaluation_filename = os.path.join(folder_name, f"evaluation_B_vs_A_user_{user_id}.json")
        with open(evaluation_filename, 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)
        raw_scores_b_vs_a.append(result)

    raw_scores_b_vs_a_filename = os.path.join(folder_name, "raw_scores_B_vs_A.json")
    with open(raw_scores_b_vs_a_filename, 'w', encoding='utf-8') as json_file:
        json.dump(raw_scores_b_vs_a, json_file, ensure_ascii=False, indent=4)
    print(f"All raw scores for B vs A have been saved to {raw_scores_b_vs_a_filename}")

    # 计算胜率
    print(f"Calculating win rates for {label} metrics...")
    win_rates = calculate_win_rate(results_a_vs_b, results_b_vs_a)

    # 保存汇总胜率数据为一个文件
    summary_filename = os.path.join(folder_name, "summary.json")
    with open(summary_filename, 'w', encoding='utf-8') as json_file:
        json.dump(win_rates, json_file, ensure_ascii=False, indent=4)
    print(f"Summary win rates have been saved to {summary_filename}")

    print(f"\nFinal Results for {label}:")
    for metric, rates in win_rates.items():
        print(f"{metric}:")
        print(f"  Traj A wins: {rates['A Wins']}")
        print(f"  Traj B wins: {rates['B Wins']}")
        print(f"  Ties: {rates['Ties']}")
        print(f"  Traj A win rate: {rates['Win Rate A']:.2%}")
        print(f"  Traj B win rate: {rates['Win Rate B']:.2%}")
        print(f"  Total: {rates['Total']}\n")

    print(f"All evaluation results have been saved in the folder: {folder_name}")

    return win_rates


def load_config(config_path: str):
    """
    从配置文件中加载参数。
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    model1 = "openai"
    model2 = "path/to/your_model_traj.log"
    # Paths to your trajectory files (可以是 .log 或 .jsonl 格式)
    traj_file_a = f"./{model1}.log"  # 修改为您的文件路径，如 "./sft.log" 或 "./sft.jsonl"
    traj_file_b = f"./{model2}.log"  # 示例：其中一个文件为 .jsonl 格式

    # Prepare comparison pairs
    print("Parsing trajectory files and preparing comparison pairs...")
    parsed_a = parse_trajectory_file(traj_file_a)
    parsed_b = parse_trajectory_file(traj_file_b)
    
    comparison_pairs = []
    for key in parsed_a:
        if key in parsed_b:
            dialogue_a = parsed_a[key][0]  # 假设每个用户只有一个对话记录
            dialogue_b = parsed_b[key][0]
            comparison_pairs.append((key, dialogue_a, dialogue_b))  # 包含 user_id
        else:
            print(f"Key {key} found in A but not in B.")

    # Load configurations
    config_path = "../../config/api_config.json"
    config = load_config(config_path)["openai"]
    openai_client = OpenAIClient(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_path=config["model_path"],
    )

    # Define evaluation format (确保花括号已转义)
    eval_format = """
    You are provided with two dialogue trajectories for comparison. Evaluate each dialogue system using the following criteria:
    1. **Flexibility**: How well does the system adapt to changes in user requests or shifts in conversation flow?
       - **1**: A is more flexible.
       - **-1**: B is more flexible.
       - **0**: Both are equally flexible.

    2. **Coherence**: How consistent and fluid is the dialogue? Does the system remember context and respond appropriately to the user's input?
       - **1**: A is more coherent.
       - **-1**: B is more coherent.
       - **0**: Both are equally coherent.

    3. **User Guidance**: How well does the system guide the user, clarify requests, or steer the conversation in a productive direction?
       - **1**: A provides better guidance.
       - **-1**: B provides better guidance.
       - **0**: Both provide similar levels of guidance.
    
    4. Refers to the above three indicators to determine which trajectory is better.

    Traj A: {Traj_a}

    Traj B: {Traj_b}

    Please provide a score of **1**, **-1**, or **0** based on the comparison. After scoring, please output the result in the following pure JSON format:

    {{
      "Flexibility": {{
        "Reason": "{{reason}}",
        "Score": -1 or 1 or 0
      }},
      "Coherence": {{
        "Reason": "{{reason}}",
        "Score": -1 or 1 or 0
      }},
      "User Guidance": {{
        "Reason": "{{reason}}",
        "Score": -1 or 1 or 0
      }},
      "Final Score": -1 or 1 or 0
    }}
    """

    # Evaluate all pairs in original order (A vs B)
    print("Evaluating dialogue pairs with OpenAI API (A vs B)...")
    results_a_vs_b = classify_comments(comparison_pairs, eval_format, openai_client, max_workers=50, swap=False)

    # Evaluate all pairs in swapped order (B vs A)
    print("Evaluating dialogue pairs with OpenAI API (B vs A)...")
    results_b_vs_a = classify_comments(comparison_pairs, eval_format, openai_client, max_workers=50, swap=True)

    # Create output folder based on model names and add timestamp
    folder_name = f"results_{model1}_vs_{model2}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Generate timestamp
    folder_path = os.path.join(folder_name, timestamp)    # Create subfolder path with timestamp

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

    # 保存 A vs B 的评估结果
    print("Saving individual evaluation results (A vs B)...")
    raw_scores_a_vs_b = []  # 用于保存所有原始评分数据
    for idx, result in enumerate(results_a_vs_b, start=1):
        if result is None:
            print(f"Skipping saving for evaluation {idx} (A vs B) due to None result.")
            continue
        raw_scores_a_vs_b.append(result)

    raw_scores_a_vs_b_filename = os.path.join(folder_path, "raw_scores_A_vs_B.json")
    with open(raw_scores_a_vs_b_filename, 'w', encoding='utf-8') as json_file:
        json.dump(raw_scores_a_vs_b, json_file, ensure_ascii=False, indent=4)
    print(f"All raw scores for A vs B have been saved to {raw_scores_a_vs_b_filename}")

    # 保存 B vs A 的评估结果
    print("Saving individual evaluation results (B vs A)...")
    raw_scores_b_vs_a = []  # 用于保存所有原始评分数据
    for idx, result in enumerate(results_b_vs_a, start=1):
        if result is None:
            print(f"Skipping saving for evaluation {idx} (B vs A) due to None result.")
            continue
        raw_scores_b_vs_a.append(result)

    raw_scores_b_vs_a_filename = os.path.join(folder_path, "raw_scores_B_vs_A.json")
    with open(raw_scores_b_vs_a_filename, 'w', encoding='utf-8') as json_file:
        json.dump(raw_scores_b_vs_a, json_file, ensure_ascii=False, indent=4)
    print(f"All raw scores for B vs A have been saved to {raw_scores_b_vs_a_filename}")

    # Calculate win rates for all metrics by averaging both evaluations
    print("Calculating averaged win rates for all metrics...")
    win_rates = calculate_win_rate(results_a_vs_b, results_b_vs_a)

    # 保存汇总胜率数据为一个文件
    summary_filename = os.path.join(folder_path, "summary.json")
    with open(summary_filename, 'w', encoding='utf-8') as json_file:
        json.dump(win_rates, json_file, ensure_ascii=False, indent=4)
    print(f"Summary win rates have been saved to {summary_filename}")

    print("\nFinal Results:")
    for metric, rates in win_rates.items():
        print(f"{metric}:")
        print(f"  Traj A wins: {rates['A Wins']}")
        print(f"  Traj B wins: {rates['B Wins']}")
        print(f"  Ties: {rates['Ties']}")
        print(f"  Traj A win rate: {rates['Win Rate A']:.2%}")
        print(f"  Traj B win rate: {rates['Win Rate B']:.2%}")
        print(f"  Total: {rates['Total']}\n")

    # 动态生成结果文件名，根据模型名称（已在汇总文件中体现）
    # 输出文件夹已包含所有相关文件

    print(f"All evaluation results have been saved in the folder: {folder_path}")


if __name__ == "__main__":
    main()
