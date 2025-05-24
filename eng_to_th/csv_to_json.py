import pandas as pd

# อ่านไฟล์ CSV
df = pd.read_csv('train_bird_th.csv')

df = df.fillna("")

# แปลงเป็น List ของ Dictionary
records = df.to_dict(orient='records')

# เซฟเป็น JSON
import json
with open('train_bird_th.json', 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)