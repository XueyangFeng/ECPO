import json

def deduplicate_items(input_file, output_file, key="title"):
    """
    根据指定的键对 JSONL 文件中的数据去重。
    
    参数:
        input_file (str): 输入文件路径 (JSONL 格式)。
        output_file (str): 输出文件路径 (JSONL 格式)。
        key (str): 用于去重的字段，默认为 "title"。
    """
    seen = set()  # 用于存储已见过的键值
    unique_items = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            item = json.loads(line.strip())  # 逐行读取并解析 JSON
            if key in item and item[key] not in seen:
                seen.add(item[key])  # 标记为已见
                unique_items.append(item)  # 添加到结果列表

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in unique_items:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')  # 写入去重后的结果

    print(f"去重完成！去重后共 {len(unique_items)} 条记录，结果已保存到 {output_file}")


# 示例使用
if __name__ == "__main__":
    input_file_path = "Yelp/items.jsonl"       # 原始 JSONL 文件路径
    output_file_path = "Yelp/unique_items.jsonl"  # 去重后的 JSONL 文件路径
    deduplicate_items(input_file_path, output_file_path, key="BusinessName")
