import json

# รายชื่อ Question ID ที่ต้องการเก็บไว้
target_ids = {
    267, 355, 1025, 1028, 1031, 1032, 1033, 1036, 1037, 1041, 1068, 1073,
    1093, 1096, 1100, 1114, 1115, 1119, 1120, 1481
}

# โหลดไฟล์ JSON
with open('bird/data/dev/dev_j2c2j.json', 'r', encoding='utf-8') as f:
    dev_data = json.load(f)

# กรองเฉพาะ Object ที่มี question_id อยู่ใน target_ids
filtered_data = [item for item in dev_data if item.get("question_id") in target_ids]

# บันทึกผลลัพธ์ลงไฟล์ใหม่
with open('bird/data/dev/dev_j2c2j_error.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print("Created successfully!!!")