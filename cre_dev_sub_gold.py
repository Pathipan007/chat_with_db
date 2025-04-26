import json

# อ่าน dev_subset.json
with open('spider_eval/dev_subset_100.json', 'r') as f:
    subset_data = json.load(f)

# สร้าง dev_gold_subset.sql
with open('spider_eval/dev_gold_subset_100.sql', 'w') as f:
    for item in subset_data:
        sql = item['query'].strip()
        db_id = item['db_id']
        f.write(f"{sql}\t{db_id}\n")

print("Created dev_gold_subset.sql successful!!!")