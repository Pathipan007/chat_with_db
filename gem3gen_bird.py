import json
import requests
import re

# ฟังก์ชันดึง schema จาก dev_tables.json (เพิ่ม backticks รอบชื่อคอลัมน์)
def get_schema(db_id, tables_file='bird/data/dev/dev_tables.json'):
    with open(tables_file, 'r') as f:
        tables = json.load(f)
    for db in tables:
        if db['db_id'] == db_id:
            schema = []
            for table_idx, table_name in enumerate(db['table_names_original']):
                columns = [f"`{col[1]}`" for col in db['column_names_original'] if col[0] == table_idx]
                if columns:
                    schema.append(f"Table: {table_name}, Columns: {', '.join(columns)}")
            return '\n'.join(schema)
    return ""

# ฟังก์ชันล้าง Markdown และแปลง SQL เป็นบรรทัดเดียว
def clean_sql(sql):
    sql = re.sub(r'```sql\n|```', '', sql, flags=re.MULTILINE)
    sql = ' '.join(sql.split())
    if not sql.endswith(';'):
        sql += ';'
    return sql.strip()

# ฟังก์ชันเรียก Ollama API
def query_ollama(prompt, error_log, question_id, question):
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
        print(f"Error for Question ID {question_id}: {e}")
        # เก็บข้อมูล error ลงใน error_log
        error_log.append({
            "question_id": question_id,
            "question": question,
            "error": str(e)
        })
        return ""

# อ่าน dev.json ของ BIRD-SQL
with open('bird/data/dev/dev.json', 'r') as f:
    dev_data = json.load(f)

# เตรียม dictionary สำหรับ predict_dev.json และ list สำหรับเก็บ error
predict_json = {}
error_log = []  # เก็บข้อมูล error

# สร้างและเก็บข้อมูลสำหรับ predict_dev.json
with open('bird/exp_result/gemma3_output/predict_dev.txt', 'w') as f:
    for i, item in enumerate(dev_data):
        question_id = item['question_id']
        question = item['question']
        db_id = item['db_id']
        schema = get_schema(db_id)
        prompt = f"""Given the following database schema:
{schema}
Translate this natural language question into a valid SQL query:
{question}
Output only the SQL query as a single line, without Markdown formatting (e.g., ```sql), explanations, or additional text."""
        sql = query_ollama(prompt, error_log, question_id, question)
        cleaned_sql = clean_sql(sql)
        # รูปแบบสำหรับ predict_dev.txt: question_id \t SQL \t----- bird -----\t db_id
        output_line = f"{question_id}\t{cleaned_sql}\t----- bird -----\t{db_id}"
        f.write(f"{output_line}\n")
        # รูปแบบสำหรับ predict_dev.json: SQL \t----- bird -----\t db_id
        json_line = f"{cleaned_sql}\t----- bird -----\t{db_id}"
        predict_json[str(question_id)] = json_line
        print(f"Processed question {i+1}/{len(dev_data)}")
        print(f"Question ID: {question_id}")
        print(f"Question: {question}")
        print(f"SQL query: {cleaned_sql}")
        print("-----------------------------------------------------------------------------------------------------------\n\n")

# สร้าง predict_dev.json
with open('bird/llm/exp_result/gemma3_output/predict_dev.json', 'w') as f:
    json.dump(predict_json, f, indent=4)

print("=== Generated successful!!! ===")

# แสดงข้อมูล error
print("\n=== Error Summary ===")
if error_log:
    print(f"Total errors: {len(error_log)}")
    for error in error_log:
        print(f"Question ID: {error['question_id']}")
        print(f"Question: {error['question']}")
        print(f"Error Message: {error['error']}")
        print("--------------------")
else:
    print("No errors occurred during generation.")