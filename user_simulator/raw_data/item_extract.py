import json

# 提取评分大于4的物品信息（不保留评分）
def extract_items(reviews_data, score):
    # 用于存储符合条件的物品数据
    high_rating_items = []

    # 遍历每个用户的数据
    for user_data in reviews_data:
        user_id = user_data.get("UserID")  # 获取用户ID
        user_reviews = user_data.get("ReviewList", [])  # 获取用户的评论列表

        # 遍历用户评论的每一条数据
        for review in user_reviews:
            rating = review.get("Stars")  # 获取评分
            if rating and float(rating) >= score:  # 评分大于4
                # 提取物品相关信息（不包含评分）
                item_data = {
                    "UserID": user_id,  # 用户ID
                    "BusinessName": review.get("BusinessName"),  # 物品名称
                    "Categories": review.get("Categories"),  # 物品类别
                    "Description": review.get("Description"),  # 物品描述
                }

                # 将符合条件的物品信息添加到结果列表中
                high_rating_items.append(item_data)

    return high_rating_items

# 加载数据的辅助函数
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

# 保存结果到JSONL文件的辅助函数
def save_to_jsonl(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for item in data:
            json.dump(item, output_file, ensure_ascii=False)
            output_file.write('\n')  # 每个JSON对象单独一行

# 主函数
def main():
    # 输入和输出文件路径
    reviews_path = "processed_data_with_summaries.jsonl"  # 用户评论数据路径
    pos_item_path = "high_rating_items.jsonl"  # 保存高分物品，用于构建用户的目标物品
    item_path = "items.jsonl" # 保存所有物品，用于构建外部数据库，供CRA检索

    # 加载用户评论数据
    reviews_data = load_jsonl(reviews_path)

    # 提取评分大于4的物品（不保留评分）
    high_rating_items = extract_items(reviews_data, score=4)
    items = extract_items(reviews_data, score=0)

    # 保存结果到JSONL文件
    save_to_jsonl(high_rating_items, pos_item_path)
    save_to_jsonl(items, item_path)

    print(f"Process completed. Results saved to {pos_item_path}")

# 执行脚本
if __name__ == "__main__":
    main()
