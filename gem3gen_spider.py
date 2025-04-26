import json
import requests
import re

# ฟังก์ชันดึง schema จาก tables.json
def get_schema(db_id, tables_file='spider/data/tables.json'):
    with open(tables_file, 'r') as f:
        tables = json.load(f)
    for db in tables:
        if db['db_id'] == db_id:
            schema = []
            for table_idx, table_name in enumerate(db['table_names_original']):
                columns = [col[1] for col in db['column_names_original'] if col[0] == table_idx]
                if columns:
                    schema.append(f"Table: {table_name}, Columns: {', '.join(columns)}")
            return '\n'.join(schema)
    return ""

# ฟังก์ชันล้าง Markdown และแปลง SQL เป็นบรรทัดเดียว
def clean_sql(sql):
    sql = re.sub(r'```sql\n|```', '', sql, flags=re.MULTILINE)
    sql = ' '.join(sql.split())
    sql = sql.rstrip(';')
    return sql.strip()

# ฟังก์ชันเรียก Ollama API
def query_ollama(prompt, error_log, index, question):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma3:12b",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['response'].strip()
    except Exception as e:
        print(f"Error at index {index+1}: {e}")
        # เก็บข้อมูล error ลงใน error_log
        error_log.append({
            "index": index+1,
            "question": question,
            "error": str(e)
        })
        return ""

# อ่าน dev.json
with open('spider/data/dev/dev.json', 'r') as f:
    dev_data = json.load(f)

# เตรียม list สำหรับเก็บ error
error_log = []

# สร้าง predicted_sql.txt
with open('spider/data/pred/gemma12b_pred_1034.txt', 'w') as f:
    for i, item in enumerate(dev_data):
        question = item['question']
        db_id = item['db_id']
        schema = get_schema(db_id)
        prompt = f"""Given the following database schema:
{schema}
Translate this natural language question into a valid SQL query:
{question}
Output only the SQL query as a single line, without Markdown formatting (e.g., ```sql), explanations, or additional text."""
        sql = query_ollama(prompt, error_log, i, question)
        # ล้าง Markdown และแปลงเป็นบรรทัดเดียว
        cleaned_sql = clean_sql(sql)
        f.write(f"{cleaned_sql}\n")
        print(f"Processed question {i+1}/{len(dev_data)}")
        print(f"Question: {question}")
        print(f"SQL query: {cleaned_sql}")
        print("-----------------------------------------------------------------------------------------------------------\n\n")

print("=== Generated successful!!! ===")

# แสดงข้อมูล error
print("\n=== Error Summary ===")
if error_log:
    print(f"Total errors: {len(error_log)}")
    for error in error_log:
        print(f"Index: {error['index']}")
        print(f"Question: {error['question']}")
        print(f"Error Message: {error['error']}")
        print("--------------------")
else:
    print("No errors occurred during generation.")