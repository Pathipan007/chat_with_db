import json

path = './bird/data/train/train_split_bird_90.json'

# อ่านไฟล์ JSON
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# เรียงข้อมูลตาม question_id
data_sorted = sorted(data, key=lambda x: x['question_id'])

# บันทึกไฟล์ใหม่
with open(path, 'w', encoding='utf-8') as f:
    json.dump(data_sorted, f, ensure_ascii=False, indent=4)

print(f"=== Save to this path: {path} ===")