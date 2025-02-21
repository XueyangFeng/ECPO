import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser
from user_simulator.user_agent_env_v1 import UserAgentEnv
from crs import ReActCRS, ActCRS, RAGCRS, ZeroShotCRS, MACRS
from datetime import datetime
from sentence_transformers import SentenceTransformer
import tiktoken

# 定义 crs_type 和类的映射
CRS_TYPES = {
    "ReActCRS": ReActCRS,
    "ActCRS": ActCRS,
    "RAGCRS": RAGCRS,
    "ZSCRS": ZeroShotCRS,
    "MACRS": MACRS,
}


CRS_LORA = {
    "ReActCRS": "react-lora",
    "ActCRS": "act-lora",
    "RAGCRS": "rag-lora",
    "ZSCRS": "zs-lora",
    "MACRS": "ma-lora",
}

def count_tokens(text, model="gpt-3.5-turbo"):
    # 加载模型对应的分词器
    encoding = tiktoken.encoding_for_model(model)
    # 计算文本的 token 数
    num_tokens = len(encoding.encode(text))
    return num_tokens

# 获取类的方法
def get_crs_class(crs_type):
    crs_class = CRS_TYPES.get(crs_type)
    if not crs_class:
        raise ValueError(f"Invalid CRS type: {crs_type}. Valid options are: {list(CRS_TYPES.keys())}")
    return crs_class


def calculate_avg_without_zeros(ratings):
    # 过滤掉所有的0值
    non_zero_ratings = [rating for rating in ratings if rating != 0]
    
    # 如果列表有非零值，计算平均值；如果没有非零值，则返回0
    if non_zero_ratings:
        avg_rating = sum(non_zero_ratings) / len(non_zero_ratings)
        return avg_rating
    else:
        return 0

def scratchpad_reward(scratchpad, rec_reward, action_reward, exp_reward, rec_reasons, action_reasons, exp_reasons):
    #print(rec_reward)
    #raise
    for count, step_data in scratchpad.items():
        try:
            # 确保 rec_reward 和 action_reward 有足够的长度
            rec_r = rec_reward[int(count)] if int(count) < len(rec_reward) else None
            action_r = action_reward[int(count)] if int(count) < len(action_reward) else None
            exp_r = exp_reward[int(count)] if int(count) < len(action_reward) else None   

            rec_reason = rec_reasons[int(count)] if int(count) < len(rec_reward) else None
            action_reason = action_reasons[int(count)] if int(count) < len(rec_reward) else None
            exp_reason = exp_reasons[int(count)] if int(count) < len(rec_reward) else None
            # 如果 rewards 数据存在，添加到 step_data 中
            step_data["rec_reward"] = rec_r
            step_data["action_reward"] = action_r
            step_data["exp_reward"] = exp_r
            step_data["rec_reason"] = rec_reason
            step_data["action_reason"] = action_reason
            step_data["exp_reason"] = exp_reason

        except (ValueError, IndexError) as e:
            print(f"Error processing count {count}: {e}")
            # 你可以选择跳过或进行其他处理
            step_data["rec_reward"] = None
            step_data["action_reward"] = None
            step_data["exp_reward"] = None
            step_data["rec_reason"] = None
            step_data["action_reason"] = None
            step_data["exp_reason"] = None
    return scratchpad



def simulate_user(crs_type, user_id, persona_path, config_path, format_path, user_model, crs_model, log_path, scratchpad_path, index_file, metadata_file, emb_model, conversation_rounds, crs_temperature, domain, apply_res_correction, query_num):
    is_correct = False
    item_id = 0


    # 用户代理环境初始化
    user_agent_env = UserAgentEnv(
        persona_path=persona_path,
        user_id=user_id,
        item_id=item_id,
        config_path=config_path,
        format_path=format_path,
        domain=domain,
        model_type=user_model
    )


    crs_class = get_crs_class(crs_type)
    crs_lora = CRS_LORA[crs_type]
    if crs_type == "MACRS":
        crs = crs_class(
            query_num=query_num,
            config_path=config_path,
            emb_model=emb_model,
            domain=domain,
            model_type=crs_model,
            index_file=index_file,
            metadata_file=metadata_file,
            format_path=format_path,
            lora = crs_lora,
            crs_temperature=crs_temperature,
            target=user_agent_env.item["ItemName"].lower()
        )
    elif crs_type == "ActCRS" or crs_type == "ReActCRS":
        crs = crs_class(
            query_num=query_num,
            config_path=config_path,
            emb_model=emb_model,
            domain=domain,
            model_type=crs_model,
            index_file=index_file,
            metadata_file=metadata_file,
            shot_type="zero_shot",
            lora = crs_lora,
            crs_temperature=crs_temperature,
            target=user_agent_env.item["ItemName"].lower()
        )
    else:
        crs = crs_class(
            query_num=query_num,
            config_path=config_path,
            emb_model=emb_model,
            domain=domain,
            model_type=crs_model,
            index_file=index_file,
            metadata_file=metadata_file,
            lora = crs_lora,
            target=user_agent_env.item["ItemName"].lower()
        )

    # 重置用户代理环境和 CRS

    user_agent_env.reset(user_id=user_id, item_id=item_id)
    crs.reset()

    rec_ratings = []
    act_ratings = []
    exp_ratings = []
    rec_reasons = []
    act_reasons = []
    exp_reasons = []
    raw_dialogue = {"Dialogue_id": f"User_id: {user_id}, Item_id: {item_id}"}

    total_user_tokens = 0
    total_system_tokens = 0
    total_rounds = 0  # 轮数计数器

    for round_num in range(conversation_rounds):
        total_rounds += 1  # 记录总轮数

        if round_num != 0:
            user_feedback = user_agent_env.step(crs_response=crs_response)
            user_input = user_feedback["user_response"]

            rec_satisfaction = json.loads(user_feedback["recommendation_satisfaction"])
            rec_rating = int(rec_satisfaction["rating"])
            rec_reason = rec_satisfaction["reason"]
            act_satisfaction = json.loads(user_feedback["action_satisfaction"])
            act_rating = int(act_satisfaction["rating"])
            act_reason = act_satisfaction["reason"]
            exp_satisfaction = json.loads(user_feedback["expression_satisfaction"])
            exp_rating = int(exp_satisfaction["rating"])
            exp_reason = exp_satisfaction["reason"]


            rec_ratings.append(rec_rating)
            act_ratings.append(act_rating)
            exp_ratings.append(exp_rating)

            rec_reasons.append(rec_reason)
            act_reasons.append(act_reason)
            exp_reasons.append(exp_reason)

            if json.loads(user_feedback["user_policy"])["policy"] == "end_conversation":
                crs.dialogue_history.add_user_message(json.loads(user_feedback["user_response"])["response"])
                raw_dialogue[f"round {round_num}"] = {"user": user_feedback}
                break
        else:
            user_feedback = user_agent_env.step()
            user_input = user_feedback["user_response"]

        # 统计用户输入的 Token 数
        user_token_count = count_tokens(user_input)
        total_user_tokens += user_token_count
        if apply_res_correction:
            if user_agent_env.item["ItemName"].lower() in user_input.lower() and not is_correct:
                raise Exception("Item name in user input")
        #TODO这里生成两个，然后下面来做simulation。对于一个user,每个step分裂一次。
        crs_response = crs.step(user_input)
        if not crs_response:
            crs_response = "Sorry, System Error"
        # 统计系统输出的 Token 数
        system_token_count = count_tokens(crs_response)
        total_system_tokens += system_token_count
        if crs_response and user_agent_env.item["ItemName"].lower() in crs_response.lower():
            is_correct = True
        raw_dialogue[f"round {round_num}"] = {"user": user_feedback, "assistant": crs_response}

    avg_rec_reward = calculate_avg_without_zeros(rec_ratings)
    avg_act_reward = calculate_avg_without_zeros(act_ratings)
    avg_exp_reward = calculate_avg_without_zeros(exp_ratings)
    is_recall = crs.get_recall()
    with open(log_path, mode="a", encoding="utf-8") as f:
        f.write(f"User {user_id}  Item {item_id}:\n")
        f.write(str(crs.dialogue_history))
        f.write("\n")
        f.write(f"rec_rewards: {rec_ratings}\n")
        f.write(f"avg_rec_reward: {avg_rec_reward}\n")
        f.write(f"action_rewards: {act_ratings}\n")
        f.write(f"avg_action_reward: {avg_act_reward}\n")
        f.write(f"expression_rewards: {exp_ratings}\n")
        f.write(f"avg_expression_reward: {avg_exp_reward}\n")
        f.write(f"Correct recommendation: {is_correct}\n")
        f.write(f"Recall: {is_recall}\n")
        f.write(f"Total user tokens: {total_user_tokens}\n")
        f.write(f"Total system tokens: {total_system_tokens}\n")
        f.write(f"Total rounds: {total_rounds}\n")
        f.write("\n" + "=" * 118 + "\n")
        f.write("\n\n")


    with open(scratchpad_path, mode="a", encoding="utf-8") as f:
            scratchpad_data = {"Dialogue_id": f"User_id: {user_id}, Item_id: {item_id}", "scratchpad": scratchpad_reward(crs.get_traj(), rec_ratings, act_ratings, exp_ratings, rec_reasons, act_reasons, exp_reasons), "rec_rewards": rec_ratings, "act_rewards": act_ratings, "is_correct": is_correct, "is_recall": is_recall}
            f.write(json.dumps(scratchpad_data) + "\n")

    return is_correct, is_recall, avg_rec_reward, avg_act_reward, avg_exp_reward, raw_dialogue, total_user_tokens, total_system_tokens, total_rounds


def main():
    parser = ArgumentParser()
    parser.add_argument("--domain", choices=["Book", "Game", "Yelp"], required=True, help="domain to test")  
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Mode for the dataset: train or test.")
    parser.add_argument("--crs_type", choices=CRS_TYPES.keys(), default="ReActCRS", help="Type of CRS to test.")
    parser.add_argument("--persona_path", default="user_simulator/persona/persona_item_v1.jsonl", help="Path to persona file.")
    parser.add_argument("--config_path", default="config/api_config.json", help="Path to config file.")
    parser.add_argument("--format_path", default="config", help="Format path for the simulator.")
    parser.add_argument("--user_model", default="openai_mini", help="User model name.")
    parser.add_argument("--crs_model", default="llama", help="CRS model name.")
    parser.add_argument("--index_file", default="data/emb/faiss_index.bin", help="Path to the FAISS index file.")
    parser.add_argument("--metadata_file", default="data/emb/metadata.json", help="Path to the metadata file.")
    parser.add_argument("--emb_model_path", default="crs/tools/all-MiniLM-L6-v2", help="Path to the embedding model.")
    parser.add_argument("--num_users", type=int, default=100, help="Number of users to simulate.")
    parser.add_argument("--num_threads", type=int, default=10, help="Number of Threads.")
    parser.add_argument("--conversation_rounds", type=int, default=5, help="Number of conversation rounds per user.")
    parser.add_argument("--output_dir", default="result_v3", help="Base output directory.")
    parser.add_argument("--crs_temperature", type=float, default=0, help="temperature of crs base model.")
    parser.add_argument("--apply_res_correction", action="store_true", help="Apply resource correction.")
    parser.add_argument("--query_num", type=int, default=5, help="Number of query results.")


    args = parser.parse_args()
    print("emb model loading")
    emb_model = SentenceTransformer(args.emb_model_path, device='cuda')
    print("emb model loaded")

    # 获取当前时间戳
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, args.crs_model, current_timestamp)
    os.makedirs(log_dir, exist_ok=True)
    raw_history_file = os.path.join(log_dir, f"raw_history_{current_timestamp}.jsonl")
    scratchpad_path = os.path.join(log_dir, f"CRS_scratchpad_{current_timestamp}.jsonl")
    log_path = os.path.join(log_dir, f"Dialogue_history_{current_timestamp}.log")

    total_correct = 0
    total_recall = 0
    avg_total_rec_reward = 0
    avg_total_act_reward = 0
    avg_total_exp_reward = 0
    total_user_tokens = 0
    total_sys_tokens = 0
    raw_histories = []
    valid_users = 0
    valid_rec_user = 0
    total_rounds = 0

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [
            executor.submit(
                simulate_user,
                args.crs_type,
                user_id,
                args.persona_path,
                args.config_path,
                args.format_path,
                args.user_model,
                args.crs_model,
                log_path,
                scratchpad_path,
                args.index_file,
                args.metadata_file,
                emb_model,
                args.conversation_rounds,
                args.crs_temperature,
                args.domain,
                args.apply_res_correction,
                args.query_num
            )
            for user_id in range(args.num_users)
        ]        
        for future in as_completed(futures):
            try:
            # 获取每个用户模拟的结果
                is_correct, is_recall, avg_rec_reward, avg_act_reward, avg_exp_reward, raw_history, user_token, sys_token, round_num = future.result()

                # 仅在任务成功的情况下进行汇总
                total_correct += int(is_correct)
                total_recall += int(is_recall)
                #avg_total_rec_reward += avg_rec_reward

                                # 仅在 avg_rec_reward 不为 0 的情况下计入
                if avg_rec_reward != 0:
                    avg_total_rec_reward += avg_rec_reward
                    valid_rec_user += 1

                avg_total_act_reward += avg_act_reward
                avg_total_exp_reward += avg_exp_reward
                total_user_tokens += user_token
                total_sys_tokens += sys_token
                total_rounds += round_num
                raw_histories.append(raw_history)

                # 统计有效用户
                valid_users += 1  # 成功完成的用户数

            except Exception as e:
                # 记录错误，但不会影响整体进程
                print(f"Error processing a user simulation (user_id): {e}")
                continue  # 跳过出错的任务，继续执行其他任务

        # for future in as_completed(futures):

        # # 获取每个用户模拟的结果
        #     is_correct, avg_rec_reward, avg_act_reward, raw_history, user_token, sys_token, round_num = future.result()

        #     # 仅在任务成功的情况下进行汇总
        #     total_correct += int(is_correct)
        #     avg_total_rec_reward += avg_rec_reward
        #     avg_total_act_reward += avg_act_reward
        #     total_user_tokens += user_token
        #     total_sys_tokens += sys_token
        #     total_rounds += round_num
        #     raw_histories.append(raw_history)

        #     # 统计有效用户
        #     valid_users += 1  # 成功完成的用户数

    # 汇总结果写入日志文件
    with open(log_path, mode="a", encoding="utf-8") as f:
        if valid_users > 0:  # 如果有有效用户，进行汇总
            f.write(f"Average user tokens across users: {total_user_tokens / valid_users:.2f}\n")
            f.write(f"Average system tokens across users: {total_sys_tokens / valid_users:.2f}\n")
            f.write(f"Average round num across users: {total_rounds / valid_users:.2f}\n")
            f.write(f"Total correct recommendations: {total_correct}/{valid_users}\n")
            f.write(f"Total recall recommendations: {total_recall}/{valid_users}\n")
            f.write(f"Average recommendation reward across users: {avg_total_rec_reward / valid_rec_user:.2f}\n")
            f.write(f"Average action reward across users: {avg_total_act_reward / valid_users:.2f}\n")
            f.write(f"Average expression reward across users: {avg_total_exp_reward / valid_users:.2f}\n")
        else:
            f.write("No valid users processed, no results to report.\n")
        
        f.write("\n" + "=" * 80 + "\n")

    with open(raw_history_file, mode="w", encoding="utf-8") as f:
        for raw_history in raw_histories:
            f.write(json.dumps(raw_history) + "\n")


if __name__ == "__main__":
    print("main")
    main()
