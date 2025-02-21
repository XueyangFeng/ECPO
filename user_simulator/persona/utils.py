import json
# 转换为自然语言描述的函数
def json_to_natural_language(data):
    result = []
    
    for category, attributes in data["seed_keys"].items():
        result.append(f"Category: {category.replace('_', ' ').title()}\n")
        
        for key, details in attributes.items():
            description = details["description"]
            examples = ", ".join(details["examples"])
            result.append(f"- {key.replace('_', ' ').title()}: {description}")
            result.append(f"  Example values: {examples}\n")
            
    return "\n".join(result)

# 从配置文件中加载参数
def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# 从文件中加载 JSON 数据
def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 从 JSONL 文件加载数据（逐行加载）
def load_jsonl(file_path: str, num_lines=None):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if num_lines is not None and i >= num_lines:
                break
            data.append(json.loads(line))
    return data



