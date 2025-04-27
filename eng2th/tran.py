import json
from deep_translator import GoogleTranslator

# โหลดไฟล์
with open('bird/data/dev/dev.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

i = 1
# แปลคำถาม
for entry in data:
    question_en = entry['question']
    evidence_en = entry['evidence']
    question_th = GoogleTranslator(source='en', target='th').translate(question_en)
    evidence_th = GoogleTranslator(source='en', target='th').translate(evidence_en)
    entry['question_th'] = question_th
    entry['evidence_th'] = evidence_th
    print(f"Process:{i}")
    i = i + 1

# บันทึกไฟล์ใหม่
with open('dev_translated.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)