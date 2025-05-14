import json
from sklearn.model_selection import train_test_split
from collections import Counter
import ast

# === Load dataset ===
with open('./bird/data/train/train_bird_th.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# === Stratified Split by db_id ===
db_ids = [item['db_id'] for item in data]  # ดึง db_id เพื่อใช้ในการแบ่ง
train_set, test_set = train_test_split(
    data,
    test_size=0.15,
    random_state=42,
    shuffle=True,
    stratify=db_ids  # Stratify ด้วย db_id
)

print(f"Total: {len(data)}")
print(f"Train: {len(train_set)}")
print(f"Test: {len(test_set)}")

# === Save split ===
with open('./bird/data/train/train_split_bird_85.json', 'w', encoding='utf-8') as f:
    json.dump(train_set, f, ensure_ascii=False, indent=4)

with open('./bird/data/train/test_split_bird_15.json', 'w', encoding='utf-8') as f:
    json.dump(test_set, f, ensure_ascii=False, indent=4)

# === Helper functions for analysis ===
def print_distribution(label, dataset, key):
    counter = Counter(item[key] for item in dataset)
    print(f"\nDistribution of '{key}' in {label}:")
    for k, v in counter.most_common(5):
        print(f"{k}: {v}")
    print(f"Unique {key}s: {len(counter)}")

def get_table_counts(dataset):
    table_counter = Counter()
    for item in dataset:
        tables = ast.literal_eval(item['table']) if isinstance(item['table'], str) else item['table']
        for table in tables:
            table_counter[table] += 1
    return table_counter

def print_table_stats(label, table_counter):
    print(f"\nTop tables in {label}:")
    for tbl, count in table_counter.most_common(5):
        print(f"{tbl}: {count}")
    print(f"Unique tables: {len(table_counter)}")

# === Analyze distributions ===
print_distribution("Train Set", train_set, "db_id")
print_distribution("Test Set", test_set, "db_id")

train_table_counts = get_table_counts(train_set)
test_table_counts = get_table_counts(test_set)

print_table_stats("Train Set", train_table_counts)
print_table_stats("Test Set", test_table_counts)

# === Check for missing tables in Test set ===
missing_tables = set(train_table_counts) - set(test_table_counts)
print(f"\nTables missing in Test set: {missing_tables}")