import json

# อ่าน dev_subset.json
with open('bird/data/train/test_split_bird_20.json', 'r') as f:
    data = json.load(f)

# สร้าง dev_gold_subset.sql
with open('bird/data/train/train_gold_test_split_20sql', 'w') as f:
    for item in data:
        sql = item['SQL'].strip()
        db_id = item['db_id']
        f.write(f"{sql}\t{db_id}\n")

print("Created Successful!!!")