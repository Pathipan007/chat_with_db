import json
import requests
import re
import time
import csv
import os

# ฟังก์ชันแปลงเวลาจากวินาทีเป็นนาที+วินาที
def format_time(seconds):
    if seconds < 0:  # กรณี error
        return "N/A"
    if seconds < 60:  # ถ้าต่ำกว่า 60 วินาที
        return f"{seconds:.2f} วินาที"
    minutes = int(seconds // 60)  # หานาที
    remaining_seconds = seconds % 60  # หาวินาทีที่เหลือ
    return f"{minutes} นาที {remaining_seconds:.2f} วินาที ({seconds:.2f} วินาที)"

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
    # วัดเวลาเริ่มต้น
    start_time = time.time()
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        # วัดเวลาสิ้นสุด
        end_time = time.time()
        generation_time = end_time - start_time  # คำนวณเวลา (วินาที)
        return response.json()['response'].strip(), generation_time
    except Exception as e:
        print(f"Error at index {index+1}: {e}")
        # เก็บข้อมูล error ลงใน error_log
        error_log.append({
            "index": index+1,
            "question": question,
            "error": str(e)
        })
        return ""

# สร้างโฟลเดอร์สำหรับเก็บ log ถ้ายังไม่มี
log_dir = 'spider/data/gemma3_output/logs/th/'
os.makedirs(log_dir, exist_ok=True)

# เตรียมไฟล์ CSV สำหรับเก็บ log (ตัดคอลัมน์ evidence ออก)
log_file = os.path.join(log_dir, '12b_log_no_v1.csv')
with open(log_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['question_num', 'question', 'generation_time_formatted', 'generation_time_seconds'])

# อ่าน dev.json
with open('spider/data/dev/dev_spider_th.json', 'r') as f:
    dev_data = json.load(f)

# เตรียม list สำหรับเก็บ error
error_log = []

# วัดเวลาเริ่มต้นทั้งหมด
overall_start_time = time.time()

# สร้าง predicted_sql.txt
with open('spider/data/pred/gemma12b_pred_1034_th.txt', 'w') as f:
    for i, item in enumerate(dev_data):
        question = item['question_th']
        db_id = item['db_id']
        schema = get_schema(db_id)
        prompt = f"""You are an expert in translating natural language questions into SQL queries. 
The questions may be in either English or Thai, and you must handle both languages correctly. 
Use the provided database schema to generate a valid SQL query. 
Interpret the question based on the schema and the meaning of the question alone.

Database schema:
{schema}

Translate the following natural language question into a valid SQL query:
{question}

### Instructions:
- Output only the SQL query as a single line.
- Do not include Markdown formatting (e.g., ```sql), explanations, or additional text."""
        sql, generation_time = query_ollama(prompt, error_log, i, question)
        # ล้าง Markdown และแปลงเป็นบรรทัดเดียว
        cleaned_sql = clean_sql(sql)

        # แปลงเวลาเป็นรูปแบบ "นาที+วินาที" ก่อนเขียนลง log
        formatted_time = format_time(generation_time)

         # บันทึก log ลงไฟล์ CSV (ตัด evidence ออก)
        with open(log_file, 'a', newline='', encoding='utf-8') as log_f:
            writer = csv.writer(log_f)
            writer.writerow([i+1, question, formatted_time, generation_time])
        
        f.write(f"{cleaned_sql}\n")
        print(f"Processed question {i+1}/{len(dev_data)}")
        print(f"Question: {question}")
        print(f"SQL query: {cleaned_sql}")
        print("-----------------------------------------------------------------------------------------------------------\n\n")

print("=== Generated successful!!! ===")

# วัดเวลาทั้งหมด
overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time
print(f"\n=== Overall Processing Time: {format_time(overall_time)} ===")

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