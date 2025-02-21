from model.model import OpenAIClient
from utils import json_to_natural_language, load_config, load_json, load_jsonl
import json
from tqdm import tqdm  # 引入 tqdm


# 构建用于生成用户画像的 prompt
def build_prompt(prompt_template: str, persona: dict, demo: dict):
    prompt = prompt_template.replace("{persona}", json.dumps(persona, ensure_ascii=False, indent=4))
    prompt = prompt.replace("{output_example_one}", json.dumps(demo["One"], ensure_ascii=False, indent=4))
    prompt = prompt.replace("{output_example_three}", json.dumps(demo["Three"], ensure_ascii=False, indent=4))
    return prompt

def rewrite_v0(config_path: str, prompt_path: str, input_path: str, output_path: str, demo_path: str, batch_size=5):
    # 加载配置
    config = load_config(config_path)["openai"]
    demo = load_json(demo_path)
    output_demo = demo
    # 加载 prompt 和 key 结构
    prompt = load_json(prompt_path)
    
    # 加载用户数据
    persona_data = load_jsonl(input_path)

    # 初始化 OpenAI 客户端
    openai_client = OpenAIClient(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_path=config["model_path"]
    )

    # 针对每个用户生成评论文本，添加 tqdm 进度条
    with open(output_path, 'w', encoding='utf-8') as output_file:
        batch_prompts = []  # 用于存储一批 prompts
        batch_user_ids = []  # 用于存储一批用户ID

        for item in tqdm(persona_data, desc="Processing Users", unit="user"):
            raw_id = item['UserID']
            raw_persona = item['FilledValues']

            
            # 构建 prompt
            rewrite_prompt_template = prompt["persona_rewrite_prompt_v2"]
            prompt_text = build_prompt(rewrite_prompt_template, raw_persona, output_demo)
            
            # 将 prompt 和用户ID添加到批处理中
            batch_prompts.append(prompt_text)
            batch_user_ids.append(raw_id)

            # 如果批次大小达到指定数量，发送请求
            if len(batch_prompts) == batch_size:
                # 批量调用 LLM 来生成填充值
                responses = openai_client.get_multi_chat_completions(user_messages=batch_prompts)

                # 解析并保存每个用户的结果
                for user_id, response in zip(batch_user_ids, responses):
                    try:
                        filled_values_json = json.loads(response)  # 尝试将响应解析为JSON
                    except json.JSONDecodeError:
                        print(f"Error: Unable to parse response for UserID {user_id} as JSON")
                        filled_values_json = response  # 若解析失败，则直接使用字符串
                    
                    # 保存每个用户的结果并立即写入文件
                    result = {"UserID": user_id, "Persona": filled_values_json}
                    json.dump(result, output_file, ensure_ascii=False)
                    output_file.write('\n')  # 每条记录写入一行

                # 清空当前批次
                batch_prompts = []
                batch_user_ids = []

        # 处理剩余不足一批的数据
        if batch_prompts:
            responses = openai_client.get_multi_chat_completions(user_messages=batch_prompts)
            for user_id, response in zip(batch_user_ids, responses):
                try:
                    filled_values_json = json.loads(response)
                except json.JSONDecodeError:
                    print(f"Error: Unable to parse response for UserID {user_id} as JSON")
                    filled_values_json = response
                
                result = {"UserID": user_id, "FilledValues": filled_values_json}
                json.dump(result, output_file, ensure_ascii=False)
                output_file.write('\n')

    return True



def rewrite(config_path: str, prompt_path: str, input_path: str, output_path: str, demo_path: str, batch_size=5):
    # 加载配置
    config = load_config(config_path)["openai"]
    demo = load_json(demo_path)
    output_demo = demo
    # 加载 prompt 和 key 结构
    prompt = load_json(prompt_path)
    
    # 加载用户数据
    persona_data = load_jsonl(input_path)

    # 初始化 OpenAI 客户端
    openai_client = OpenAIClient(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_path=config["model_path"]
    )

    # 针对每个用户生成评论文本，添加 tqdm 进度条
    with open(output_path, 'w', encoding='utf-8') as output_file:
        batch_prompts = []  # 用于存储一批 prompts
        batch_user_ids = []  # 用于存储一批用户ID

        for item in tqdm(persona_data, desc="Processing Users", unit="user"):
            raw_id = item['UserID']
            raw_persona = item['FilledValues']

            # 构建 prompt
            rewrite_prompt_template = prompt["persona_rewrite_prompt_v4"]
            prompt_text = build_prompt(rewrite_prompt_template, raw_persona, output_demo)
            
            # 将 prompt 和用户ID添加到批处理中
            batch_prompts.append(prompt_text)
            batch_user_ids.append(raw_id)

            # 如果批次大小达到指定数量，发送请求
            if len(batch_prompts) == batch_size:
                # 批量调用 LLM 来生成填充值
                responses = openai_client.get_multi_chat_completions(user_messages=batch_prompts)

                # 解析并保存每个用户的结果
                for user_id, response in zip(batch_user_ids, responses):
                    process_response(user_id, response, output_file)

                # 清空当前批次
                batch_prompts = []
                batch_user_ids = []

        # 处理剩余不足一批的数据
        if batch_prompts:
            responses = openai_client.get_multi_chat_completions(user_messages=batch_prompts)
            for user_id, response in zip(batch_user_ids, responses):
                process_response(user_id, response, output_file)

    return True


def process_response(user_id, response, output_file):
    """
    处理单个 LLM 响应，将其格式化为符合要求的 JSON 格式。
    """
    try:
        # 尝试将响应解析为 JSON
        filled_values_json = json.loads(response)

        # 检查是否包含多个 Personas
        personas = filled_values_json.get("Personas", [])
        if not personas:
            # 如果没有 Personas 键，视为单一 Persona
            personas = [filled_values_json]

        # 格式化并写入文件
        for persona in personas:
            try:
                formatted_entry = {
                    "UserID": user_id,
                    "Persona": {
                        "Activities": persona["Activities"],
                        "Linguistics": {
                            "Information Density": persona["Linguistics"]["Information Density"],
                            "Expression Style": persona["Linguistics"]["Expression Style"],
                            "Tone": persona["Linguistics"]["Tone"]
                        }
                    }
                }                
                json.dump(formatted_entry, output_file, ensure_ascii=False)
                output_file.write('\n')
            except:
                continue
    except json.JSONDecodeError:
        print(f"Error: Unable to parse response for UserID {user_id} as JSON")
        error_entry = {"UserID": user_id, "Error": response}
        json.dump(error_entry, output_file, ensure_ascii=False)
        output_file.write('\n')

# 示例：外部文件中调用
if __name__ == "__main__":
    # 设定路径
    config_path = "../../config/api_config.json"
    prompt_path = "../prompts/prompts.json"
    input_path = "personas_yelp.jsonl"
    demo_path = "demo.json"
    output_path = "personas_crs_yelp.jsonl"

    # 执行填充值生成并保存
    rewrite(config_path, prompt_path, input_path, output_path, demo_path, batch_size=10)
    print("Process completed.")
