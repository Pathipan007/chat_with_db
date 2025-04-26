import pandas as pd

# โหลดไฟล์ .xlsx
df = pd.read_csv('trancsv.csv')

# แปลงเป็น JSON และเซฟด้วย indent=2
df.to_json('output_2.json', orient='records', force_ascii=False, indent=2)
