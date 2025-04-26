import json
import csv

# โหลดไฟล์ JSON
with open('bird/data/dev/dev.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# ถ้าไฟล์เป็น List ของ Objects
if isinstance(data, list):
    with open('dev.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
else:
    print("ไม่ใช่ List ของ Object นะ ต้องแปลงก่อน")