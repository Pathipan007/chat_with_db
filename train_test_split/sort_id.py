import json

path = './bird/data/train/test_split_bird_20.json'

# อ่านไฟล์ JSON
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# เรียงข้อมูลตาม question_id
data_sorted = sorted(data, key=lambda x: x['question_id'])

# บันทึกไฟล์ใหม่
with open(path, 'w', encoding='utf-8') as f:
    json.dump(data_sorted, f, ensure_ascii=False, indent=4)

print(f"=== Save to this path: {path} ===")