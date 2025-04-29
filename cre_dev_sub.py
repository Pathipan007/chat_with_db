import json

# อ่าน dev.json
with open('bird/data/dev/dev_j2c2j.json', 'r', encoding='utf-8') as f:
    dev_data = json.load(f)

# เลือก 100 แถวแรก
subset_data = dev_data[:100]

# บันทึกเป็น dev_j2c2j_100.json
with open('bird/data/dev/dev_j2c2j_100.json', 'w', encoding='utf-8') as f:
    json.dump(subset_data, f, ensure_ascii=False, indent=2)

print("Created dev_j2c2j_100.json successfully!!!")