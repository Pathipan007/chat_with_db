import json
import pandas as pd

with open('bird/data/dev/dev.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.to_csv('dev_questions.csv', index=False, encoding='utf-8-sig')