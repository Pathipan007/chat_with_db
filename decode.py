import jsonlines
import json

# อ่านไฟล์ JSON แบบ jsonlines
with jsonlines.open('output.json') as reader:
    data = [obj for obj in reader]

# แปลงข้อความ Unicode เป็นข้อความที่สามารถอ่านได้
for item in data:
    if item.get('question_th'):  # ตรวจสอบว่ามีค่าใน 'question_th' หรือไม่
        item['question_th'] = item['question_th'].encode().decode('unicode_escape')
    if item.get('evidence_th') and item['evidence_th'] is not None:  # ตรวจสอบว่ามีค่าใน 'evidence_th' หรือไม่ และไม่เป็น None
        item['evidence_th'] = item['evidence_th'].encode().decode('unicode_escape')

# บันทึกข้อมูลที่จัดรูปแบบแล้วลงในไฟล์ใหม่
with open('formatted_output.json', 'w', encoding='utf-8') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=2)

print("ข้อมูล JSON ได้ถูกบันทึกลงไฟล์ 'formatted_output.json' แล้ว")
