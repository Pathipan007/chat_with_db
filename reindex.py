import json

# โหลดไฟล์ JSON
with open('bird/data/train/train_split_bird_80.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# รีเซ็ต question_id ให้เรียงลำดับใหม่
for i, item in enumerate(data, start=1):
    item['question_id'] = i

# เซฟกลับลงไฟล์ใหม่ (หรือจะทับไฟล์เดิมก็ได้)
with open('bird/data/train/train_split_bird_80_reindex.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
