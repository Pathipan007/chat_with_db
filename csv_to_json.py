import pandas as pd

# อ่านไฟล์ CSV
df = pd.read_csv('bird_tran.csv')

df = df.fillna("")

# แปลงเป็น List ของ Dictionary
records = df.to_dict(orient='records')

# เซฟเป็น JSON
import json
with open('bird_tran.json', 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)