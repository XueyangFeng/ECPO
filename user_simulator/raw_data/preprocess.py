import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import rcParams

# Ensure proper display of negative signs in matplotlib
rcParams['axes.unicode_minus'] = False


# =============================== File Handling Module ===============================

def read_jsonl_file(file_path, n=5):
    """
    Reads the first `n` entries from a JSONL file.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def load_json(file_path: str):
    """
    Loads JSON data from a file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def save_json(data, file_path):
    """
    Saves data to a JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {file_path}")


# =============================== Data Analysis & Statistics Module ===============================

def count_user_reviews(file_path):
    """
    Counts the number of reviews for each `user_id`.
    """
    user_counts = defaultdict(int)

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Processing reviews", unit="line"):
            record = json.loads(line)
            user_id = record.get('user_id')
            if user_id:
                user_counts[user_id] += 1

    return user_counts


def plot_user_comment_distribution(user_counts, output_image_path):
    """
    Plots the distribution of review counts per user and saves the figure.
    """
    bins = [0, 1, 5, 10, 20, 30, 40, 50, float('inf')]  # Review count intervals
    labels = ['1', '2-5', '5-10', '10-20', '20-30', '30-40', '40-50', '50+']
    user_distribution = defaultdict(int)

    # Categorize users into bins
    for count in user_counts.values():
        for i in range(len(bins) - 1):
            if bins[i] < count <= bins[i + 1]:
                user_distribution[labels[i]] += 1
                break

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, [user_distribution[label] for label in labels], color='skyblue')
    plt.xlabel('Number of Comments')
    plt.ylabel('Number of Users')
    plt.title('User Comment Distribution')

    # Add text labels to bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.show()


# =============================== Data Processing Module ===============================

def filter_users_by_review_count(user_counts, review_count):
    """
    Filters users who have a specific number of reviews.
    """
    return [user_id for user_id, count in user_counts.items() if count == review_count]


def extract_reviews_for_users(review_file_path, user_ids, output_file_path):
    """
    Extracts reviews from selected users and saves them to a file.
    """
    user_reviews = {user_id: [] for user_id in user_ids}

    with open(review_file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Extracting Reviews", unit="line"):
            record = json.loads(line)
            user_id = record.get('user_id')

            if user_id in user_ids:
                user_reviews[user_id].append(record)

    save_json(user_reviews, output_file_path)


# =============================== Business Metadata Processing Module ===============================

def index_meta_data(meta_file_path):
    """
    Indexes business metadata and returns a dictionary.
    """
    meta_index = {}

    with open(meta_file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Indexing Meta Data", unit="line"):
            item = json.loads(line)
            business_id = item.get('business_id')
            if business_id:
                meta_index[business_id] = item

    return meta_index


def match_reviews_with_meta(review_file_path, meta_index):
    """
    Matches reviews with business metadata.
    """
    matched_reviews = []

    with open(review_file_path, 'r', encoding='utf-8') as review_file:
        for line in tqdm(review_file, desc="Matching Reviews with Meta", unit="line"):
            review = json.loads(line)
            business_id = review.get('business_id')

            if business_id in meta_index:
                matched_reviews.append({
                    "review": review,
                    "meta": meta_index[business_id]
                })

    return matched_reviews


def load_top_n_users(user_file_path, n=100):
    """
    Loads the top `n` user IDs from a file.
    """
    with open(user_file_path, 'r', encoding='utf-8') as file:
        all_users = json.load(file)

    return all_users[:n]  # Return the first `n` users


def process_data(input_file, output_file):
    """
    Processes extracted review data and saves it in JSONL format.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        original_data = json.load(file)

    with open(output_file, 'w', encoding='utf-8') as file:
        for user_id, reviews in original_data.items():
            user_data = {
                "UserID": user_id,
                "ReviewList": []
            }

            for entry in reviews:
                review = entry["review"]
                meta = entry["meta"]
                # Please replace the mapping according to the dataset
                combined_entry = {
                    "ID": review["review_id"],
                    "Stars": review["stars"],
                    "Content": review["text"],
                    "BusinessID": review["business_id"],
                    "BusinessName": meta["name"],
                    "Rating": meta.get("rating", 0.0),
                    "Address": meta.get("address", ""),
                    "City": meta.get("city", ""),
                    "State": meta.get("state", ""),
                    "PostalCode": meta.get("postal_code", ""),
                    "Categories": meta.get("categories", "").split(", "),
                    "Attributes": meta.get("attributes", {}),
                    "Hours": meta.get("hours", {}),
                }

                user_data["ReviewList"].append(combined_entry)

            file.write(json.dumps(user_data, ensure_ascii=False) + '\n')

    print(f"Processed data saved in {output_file} (JSONL format).")


# =============================== Main Execution ===============================

def main():
    # File path placeholders
    review_file_path = 'path/to/review_data.jsonl'  # User review dataset
    meta_file_path = 'path/to/business_metadata.jsonl'  # Business metadata dataset
    user_counts_file = 'path/to/user_review_counts.json'  # Output: User review counts
    filtered_users_file = 'path/to/filtered_users.json'  # Output: Filtered users (e.g., 30 reviews)
    processed_data_output = 'path/to/processed_reviews.jsonl'  # Output: Final processed data
    index_file = "path/to/meta_index.json"  # Output: Business metadata index

    user_counts = count_user_reviews(review_file_path)
    save_json(user_counts, user_counts_file)

    filtered_users = filter_users_by_review_count(user_counts, 30)
    save_json(filtered_users, filtered_users_file)

    extract_reviews_for_users(review_file_path, filtered_users, 'path/to/user_reviews.jsonl')

    meta_index = index_meta_data(meta_file_path)
    save_json(meta_index, index_file)

    matched_reviews = match_reviews_with_meta(review_file_path, meta_index)
    save_json(matched_reviews, 'path/to/matched_reviews.jsonl')

    process_data('path/to/matched_reviews.json', processed_data_output)


if __name__ == "__main__":
    main()
