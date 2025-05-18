import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import ast

# === Load dataset ===
with open('./bird/data/train/train_smote_class2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# === Stratified Split by db_id ===
diff = [item['difficulty'] for item in data]  # ดึง db_id เพื่อใช้ในการแบ่ง
train_set, test_set = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=diff
)

print(f"Total: {len(data)}")
print(f"Train: {len(train_set)}")
print(f"Test: {len(test_set)}")

# === Save split ===
with open('./bird/data/train/train_split_bird_80.json', 'w', encoding='utf-8') as f:
    json.dump(train_set, f, ensure_ascii=False, indent=4)

with open('./bird/data/train/test_split_bird_20.json', 'w', encoding='utf-8') as f:
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

def print_distribution_percentage(label, dataset, key):
    counter = Counter(item[key] for item in dataset)
    total = sum(counter.values())
    print(f"\nPercentage Distribution of '{key}' in {label}:")
    for k, v in counter.most_common():
        percent = (v / total) * 100
        print(f"{k}: {v} ({percent:.2f}%)")

def print_table_stats(label, table_counter):
    print(f"\nTop tables in {label}:")
    for tbl, count in table_counter.most_common(5):
        print(f"{tbl}: {count}")
    print(f"Unique tables: {len(table_counter)}")

def plot_difficulty_percentage(train_set, test_set):
    # นับจำนวน
    train_counter = Counter(item['difficulty'] for item in train_set)
    test_counter = Counter(item['difficulty'] for item in test_set)

    # หารเปอร์เซ็นต์
    train_total = sum(train_counter.values())
    test_total = sum(test_counter.values())

    labels = sorted(set(train_counter.keys()).union(set(test_counter.keys())))
    train_percent = [(train_counter.get(label, 0) / train_total) * 100 for label in labels]
    test_percent = [(test_counter.get(label, 0) / test_total) * 100 for label in labels]

    x = range(len(labels))
    width = 0.35

    # วาดกราฟ
    fig, ax = plt.subplots()
    ax.bar([i - width/2 for i in x], train_percent, width, label='Train')
    ax.bar([i + width/2 for i in x], test_percent, width, label='Test')

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Percentage Distribution of Difficulty in Train and Test Sets')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# === Analyze distributions ===
print_distribution("Train Set", train_set, "db_id")
print_distribution("Test Set", test_set, "db_id")

print_distribution("Train Set", train_set, "difficulty")
print_distribution("Test Set", test_set, "difficulty")

print_distribution_percentage("Train Set", train_set, "difficulty")
print_distribution_percentage("Test Set", test_set, "difficulty")

# plot_difficulty_percentage(train_set, test_set)

train_table_counts = get_table_counts(train_set)
test_table_counts = get_table_counts(test_set)

print_table_stats("Train Set", train_table_counts)
print_table_stats("Test Set", test_table_counts)

# === Check for missing tables in Test set ===
# missing_tables = set(train_table_counts) - set(test_table_counts)
# print(f"\nTables missing in Test set: {missing_tables}")