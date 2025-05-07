import json
import os
import re
from sqlglot import parse_one
from sqlglot.expressions import Table

def sanitize_sql(sql):
    # ลบ backticks
    sql = sql.replace("`", "")
    
    # Escape column names ที่ใช้เครื่องหมายแปลก เช่น +/- ให้เป็นชื่อปกติชั่วคราว
    sql = re.sub(r'\b([a-zA-Z0-9_]+)\.\+/\-', r'\1.plus_minus', sql)
    return sql

def extract_tables_from_sql(sql):
    try:
        cleaned_sql = sanitize_sql(sql)
        expression = parse_one(cleaned_sql, error_level='IGNORE')
        tables = expression.find_all(Table)
        return sorted({table.name for table in tables})
    except Exception as e:
        print(f"Error parsing SQL at index: {idx}")
        print(f"Error parsing SQL: {e}")
        print(f"SQL Query: {sql}")  # พิมพ์ SQL query ที่ทำให้เกิด error
        print("=================================")
        return []

# อ่านไฟล์ train.json
train_file = 'bird/data/train/train.json'
with open(train_file, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# ประมวลผลแต่ละคำถามเพื่อเพิ่มฟิลด์ question_id และ tables
for idx, item in enumerate(train_data, start=1):
    print(f"Processing item {idx}...")

    # เพิ่ม question_id ถ้ายังไม่มี
    if 'question_id' not in item:
        item['question_id'] = idx
        print(f"  Added question_id: {idx}")
    elif not item['question_id']:  # ถ้ามีแต่เป็นค่าว่าง
        item['question_id'] = idx
        print(f"  Replaced empty question_id with: {idx}")
    else:
        print(f"  question_id already exists: {item['question_id']}")

    # ดึงชื่อตารางจาก SQL และเพิ่มฟิลด์ tables
    sql = item.get('SQL', '')
    if sql:
        table = extract_tables_from_sql(sql)
        item['table'] = table
        print(f"  Extracted tables from SQL: {table}")
    else:
        item['table'] = []
        print(f"  No SQL found. Set tables to empty list.")

# สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
output_dir = 'bird/data/train/'
os.makedirs(output_dir, exist_ok=True)

# บันทึกไฟล์ใหม่
output_file = os.path.join(output_dir, 'train_v2.json')
with open(output_file, 'w', encoding='utf-8') as f:
    # จัดเรียง question_id ไว้ลำดับแรก
    for item in train_data:
        if 'question_id' in item:
            ordered_item = {'question_id': item['question_id']}
            for k, v in item.items():
                if k != 'question_id':
                    ordered_item[k] = v
            item.clear()
            item.update(ordered_item)
    json.dump(train_data, f, ensure_ascii=False, indent=4)

print(f"Processed {len(train_data)} questions and saved to {output_file}")