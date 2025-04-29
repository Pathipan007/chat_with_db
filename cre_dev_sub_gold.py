import json

# อ่าน dev_subset.json
with open('bird/data/dev/dev.json', 'r') as f:
    dev_data = json.load(f)

subset_data = dev_data[:100]

# สร้าง dev_gold_subset.sql
with open('bird/data/dev/dev_gold_100.sql', 'w') as f:
    for item in subset_data:
        sql = item['SQL'].strip()
        db_id = item['db_id']
        f.write(f"{sql}\t{db_id}\n")

print("Created dev_gold_subset.sql successful!!!")