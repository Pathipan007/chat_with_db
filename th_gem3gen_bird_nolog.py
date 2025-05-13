import json
import requests
import re
import time

# ฟังก์ชันแปลงเวลาจากวินาทีเป็นนาที+วินาที
def format_time(seconds):
    if seconds < 0:  # กรณี error
        return "N/A"
    if seconds < 60:  # ถ้าต่ำกว่า 60 วินาที
        return f"{seconds:.2f} วินาที"
    minutes = int(seconds // 60)  # หานาที
    remaining_seconds = seconds % 60  # หาวินาทีที่เหลือ
    return f"{minutes} นาที {remaining_seconds:.2f} วินาที ({seconds:.2f} วินาที)"

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

# ฟังก์ชันเรียก Ollama API และวัดเวลา
def query_ollama(prompt, error_log, question_id, question):
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
        print(f"Error for Question ID {question_id}: {e}")
        error_log.append({
            "question_id": question_id,
            "question": question,
            "error": str(e)
        })
        # ถ้ามี error ให้คืนค่า generation_time เป็น -1
        return "", -1

# อ่าน dev.json ของ BIRD-SQL
with open('bird/data/dev/dev_j2c2j.json', 'r', encoding='utf-8') as f:
    dev_data = json.load(f)

# เตรียม dictionary สำหรับ predict_dev.json และ list สำหรับเก็บ error
predict_json = {}
error_log = []

# วัดเวลาเริ่มต้นทั้งหมด
overall_start_time = time.time()

# ประมวลผลและเก็บข้อมูลสำหรับ predict_dev.json
for i, item in enumerate(dev_data):
    question_id = item['question_id']
    question = item['question_th']
    db_id = item['db_id']
    difficulty = item.get('difficulty')  # ดึง difficulty ถ้าไม่มีให้เป็น N/A
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
        
    # เรียก API และเก็บเวลาการ Generate
    sql, generation_time = query_ollama(prompt, error_log, question_id, question)
    cleaned_sql = clean_sql(sql)
    
    # แปลงเวลาเป็นรูปแบบ "นาที+วินาที" ก่อนเขียนลง log
    formatted_time = format_time(generation_time)
    
    # รูปแบบสำหรับ predict_dev.json: SQL \t----- bird -----\t db_id
    json_line = f"{cleaned_sql}\t----- bird -----\t{db_id}"
    predict_json[str(question_id)] = json_line
    
    print(f"Processed question {i+1}/{len(dev_data)}")
    print(f"Question ID: {question_id}")
    print(f"Question: {question}")
    print(f"Difficulty: {difficulty}")
    print(f"SQL query: {cleaned_sql}")
    print(f"Generation Time: {formatted_time}")
    print("-----------------------------------------------------------------------------------------------------------\n\n")

# สร้าง predict_dev.json
with open('bird/exp_result/gemma3_output/th/predict_dev.json', 'w', encoding='utf-8') as f:
    json.dump(predict_json, f, ensure_ascii=False, indent=4)

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
        print(f"Question ID: {error['question_id']}")
        print(f"Question: {error['question']}")
        print(f"Error Message: {error['error']}")
        print("--------------------")
else:
    print("No errors occurred during generation.")