import sys
import json
from tqdm import tqdm  # 进度条
from model.model import OpenAIClient
from utils import json_to_natural_language, load_config, load_json, load_jsonl

# 将用户的多条评论拼接为自然语言文本
def extract_and_combine_reviews(user_reviews_data):
    combined_reviews = []
    for review in user_reviews_data:
        title = review.get("Title", "No Title")
        content = review.get("Content", "No Content")
        rating = review.get("Rating", "No Rating")
        item_name = review.get("ItemName", "No Item Name")
        categories = ", ".join(review.get("Categories", []))
        author_name = review.get("AuthorName", "Unknown Author")
        price = review.get("Price", "No Price")
        description = review.get("Description", "No Description")
        
        review_text = (
            f"Title: {title}\n"
            f"ReviewContent: {content}\n"
            f"Item: {item_name}\n"
            f"Author: {author_name}\n"
            f"Rating: {rating}\n"
            f"ItemPrice: {price}\n"
            f"ItemCategories: {categories}\n"
            f"ItemDescription: {description}\n"
        )
        combined_reviews.append(review_text)
    
    return "\n\n".join(combined_reviews)

# 构建用于生成用户画像的 prompt
def build_prompt(prompt_template: str, user_reviews: str, key_structure: dict, output_example: dict):
    prompt = prompt_template.replace("{user_reviews}", user_reviews)
    prompt = prompt.replace("{domain}", "Book")
    prompt = prompt.replace("{key_structure}", json.dumps(key_structure, ensure_ascii=False, indent=4))
    prompt = prompt.replace("{output_example}", json.dumps(output_example, ensure_ascii=False, indent=4))
    return prompt

# 执行填充值生成
def fill_value(config_path: str, prompt_path: str, key_path: str, reviews_path: str, output_path: str, demo_path: str, batch_size=5):
    config = load_config(config_path)["openai"]
    demo = load_json(demo_path)["Yelp"]
    prompt = load_json(prompt_path)
    key_structure = load_json(key_path)["Yelp"]
    reviews_data = load_jsonl(reviews_path)

    openai_client = OpenAIClient(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_path=config["model_path"]
    )

    with open(output_path, 'w', encoding='utf-8') as output_file:
        batch_prompts = []
        batch_user_ids = []
        
        for user_data in tqdm(reviews_data, desc="Processing Users", unit="user"):
            user_reviews = user_data["ReviewList"]
            user_id = user_data["UserID"]
            reviews_text = extract_and_combine_reviews(user_reviews)

            value_prompt_template = prompt["domain_value_prompt_v0"]
            output_example = demo
            prompt_text = build_prompt(value_prompt_template, reviews_text, key_structure, output_example)

            batch_prompts.append(prompt_text)
            batch_user_ids.append(user_id)

            if len(batch_prompts) == batch_size:
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

                batch_prompts = []
                batch_user_ids = []

        if batch_prompts:
            responses = openai_client.get_single_chat_completions(batch_prompts)
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

# 移除嵌套 JSONL 中的指定 key
def remove_nested_key_from_jsonl(input_file, output_file, keys_to_remove):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                try:
                    json_obj = json.loads(line)
                    
                    temp = json_obj
                    for key in keys_to_remove[:-1]:
                        if key in temp:
                            temp = temp[key]
                        else:
                            break
                    if keys_to_remove[-1] in temp:
                        del temp[keys_to_remove[-1]]

                    json.dump(json_obj, outfile, ensure_ascii=False)
                    outfile.write('\n')
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON format in line: {line}")
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# 主程序入口
if __name__ == "__main__":
    # 配置路径
    config_path = "../../config/api_config.json"
    prompt_path = "../prompts/prompts.json"
    reviews_path = "../../raw_data/Yelp/processed_data_with_summaries.jsonl"
    key_path = "domain_keys.json"
    output_path = "personas_gen.jsonl"  # 初步生成的用户画像
    filtered_output_path = "personas_yelp.jsonl"  # 处理后的用户画像
    demo_path = "../demo/demo.json"

    # 生成填充值
    print("Starting value filling process...")
    fill_value(config_path, prompt_path, key_path, reviews_path, output_path, demo_path, batch_size=10)
    print("Value filling completed.")

    # 移除指定 key
    keys_to_remove = ["FilledValues", "Behavioral Traits", "Emotional Tone"]
    print("Starting key removal process...")
    remove_nested_key_from_jsonl(output_path, filtered_output_path, keys_to_remove)
    print("Key removal completed. Process finished.")
