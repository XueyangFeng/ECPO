import json

# 加载 JSONL 文件
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

# 直接合并 persona 和 items，并拆分成训练 & 测试集
def merge_and_split(persona_data, item_data, test_file, train_file):
    item_map = {}

    # 按 `UserID` 聚合物品数据
    for item in item_data:
        user_id = item.get('UserID')
        if user_id:
            if user_id not in item_map:
                item_map[user_id] = []
            item_map[user_id].append({
                "ItemName": item.get("BusinessName"),
                "Categories": item.get("Categories"),
                "Description": item.get("Description")
            })

    # 直接拆分成训练和测试集
    with open(test_file, 'w', encoding='utf-8') as test_outfile, \
         open(train_file, 'w', encoding='utf-8') as train_outfile:

        for persona in persona_data:
            user_id = persona['UserID']
            items = item_map.get(user_id, [])

            if not items:
                continue  # 跳过没有物品数据的用户

            # 生成测试集：只包含第一个 item
            test_entry = {
                "UserID": user_id,
                "Persona": persona["Persona"],  # 填充用户画像
                "Items": [items[0]]  # 只保留第一个 item
            }
            test_outfile.write(json.dumps(test_entry, ensure_ascii=False) + '\n')

            # 生成训练集：其余 items，每个 item 作为单独记录
            for item in items[1:]:
                train_entry = {
                    "UserID": user_id,
                    "Persona": persona["Persona"],
                    "Items": [item]  # 每个 item 独立存储
                }
                train_outfile.write(json.dumps(train_entry, ensure_ascii=False) + '\n')

    print(f"合并 & 拆分完成！测试集: {test_file}，训练集: {train_file}")

# 主函数
def main():
    # 输入文件路径
    persona_file_path = 'personas_crs_yelp.jsonl'  # 用户画像数据
    item_file_path = '../../raw_data/Yelp/high_rating_items.jsonl'  # 评分高的物品数据
    test_file_path = 'Yelp_test.jsonl'    # 测试集
    train_file_path = 'Yelp_train.jsonl'  # 训练集

    # 加载数据
    persona_data = load_jsonl(persona_file_path)
    item_data = load_jsonl(item_file_path)

    # 合并数据并拆分
    merge_and_split(persona_data, item_data, test_file_path, train_file_path)

# 执行主程序
if __name__ == "__main__":
    main()
