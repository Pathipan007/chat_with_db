import json
import requests
import re
import time
import csv
import os

# ฟังก์ชันแปลงเวลาจากวินาทีเป็นนาที+วินาที
def format_time(seconds):
    if seconds < 0:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.2f} วินาที"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes} นาที {remaining_seconds:.2f} วินาที ({seconds:.2f} วินาที)"

# ฟังก์ชันเรียก Ollama API และวัดเวลา
def query_ollama(prompt, error_log, question_id, context, max_retries=3, timeout=120):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma3:12b",
        "prompt": prompt,
        "stream": False
    }
    start_time = time.time()
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            end_time = time.time()
            generation_time = end_time - start_time
            return response.json()['response'].strip(), generation_time
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error for Question ID {question_id} after {max_retries} attempts: {e}")
                error_log.append({
                    "question_id": question_id,
                    "context": context,
                    "error": str(e)
                })
                return "", -1
            time.sleep(1)
    return "", -1

# ฟังก์ชันแปลภาษาด้วย LLM และเก็บเวลาแปล
def translate_to_english_with_llm(text, error_log, question_id):
    if not text:
        return "", -1
    prompt = f"""You are an expert translator. Translate the following Thai text into English. Use terminology suitable for a database context (e.g., translate "ใบแจ้งยอด" as "statement" instead of "invoice", "รายสัปดาห์" as "weekly" instead of "per week").

Thai text:
{text}

### Instructions:
- Output only the translated English text.
- Do not include explanations or additional text.
"""
    translated_text, translation_time = query_ollama(prompt, error_log, question_id, f"Translation of: {text}")
    translated_text = translated_text if translated_text else text
    return translated_text, translation_time

# ฟังก์ชันดึง schema จาก dev_tables.json
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

# สร้างโฟลเดอร์สำหรับเก็บ log
log_dir = 'bird/exp_result/gemma3_output_kg/logs/th/'
os.makedirs(log_dir, exist_ok=True)

# เตรียมไฟล์ CSV สำหรับเก็บ log
log_file = os.path.join(log_dir, '12b_log_envi_j2c2j_tran_error.csv')
with open(log_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['question_id', 'question_th', 'question_en', 'question_translation_time', 'evidence_th', 'evidence_en', 'evidence_translation_time', 'difficulty', 'generated_sql', 'gold_sql', 'generation_time_formatted', 'generation_time_seconds'])

# อ่าน dev.json ของ BIRD-SQL
with open('bird/data/dev/dev_j2c2j_error.json', 'r', encoding='utf-8') as f:
    dev_data = json.load(f)

# เตรียม dictionary สำหรับ predict_dev.json และ list สำหรับเก็บ error
predict_json = {}
error_log = []

# วัดเวลาเริ่มต้นทั้งหมด
overall_start_time = time.time()

# ประมวลผลและเก็บข้อมูลสำหรับ predict_dev.json
for i, item in enumerate(dev_data):
    question_id = item['question_id']
    question_th = item['question_th']
    db_id = item['db_id']
    evidence_th = item.get('evidence_th', '')
    difficulty = item.get('difficulty')
    gold_sql = item.get('SQL', 'N/A')
    schema = get_schema(db_id)
    
    # แปลคำถามและ evidence เป็นภาษาอังกฤษด้วย LLM พร้อมเก็บเวลา
    question_en, question_translation_time = translate_to_english_with_llm(question_th, error_log, question_id)
    evidence_en, evidence_translation_time = translate_to_english_with_llm(evidence_th, error_log, question_id)
    evidence_text = f"\nAdditional evidence: {evidence_en}" if evidence_en else ""
    
    # Prompt สำหรับ Generate SQL (ใช้คำถามภาษาอังกฤษ)
    prompt = f"""You are an expert in translating natural language questions into SQL queries. 
Use the provided database schema to generate a valid SQL query. 
If evidence is provided, use it to help interpret the question. 
If no evidence is provided, interpret the question based on the schema and the meaning of the question alone.

**Important:** All values in the database (e.g., column values, conditions) are in English. Use English values in the SQL query.

Database schema:
{schema}

Evidence (optional, use this to interpret the question if provided):
{evidence_text}

Translate the following natural language question into a valid SQL query:
{question_en}

### Instructions:
- Output only the SQL query as a single line.
- Do not include Markdown formatting (e.g., ```sql), explanations, or additional text.
"""
        
    # เรียก API และเก็บเวลาการ Generate
    sql, generation_time = query_ollama(prompt, error_log, question_id, question_th, max_retries=3)
    cleaned_sql = clean_sql(sql)
    
    # แปลงเวลาเป็นรูปแบบ "นาที+วินาที"
    formatted_generation_time = format_time(generation_time)
    formatted_question_translation_time = format_time(question_translation_time)
    formatted_evidence_translation_time = format_time(evidence_translation_time)
    
    # บันทึก log (เก็บทั้งคำถามภาษาไทยและอังกฤษ รวมถึงเวลาแปล)
    with open(log_file, 'a', newline='', encoding='utf-8') as log_f:
        writer = csv.writer(log_f)
        writer.writerow([question_id, question_th, question_en, question_translation_time, evidence_th, evidence_en, evidence_translation_time, difficulty, cleaned_sql, gold_sql, formatted_generation_time, generation_time])
    
    # รูปแบบสำหรับ predict_dev.json
    json_line = f"{cleaned_sql}\t----- bird -----\t{db_id}"
    predict_json[str(question_id)] = json_line
    
    print(f"Processed question {i+1}/{len(dev_data)}")
    print(f"Question ID: {question_id}")
    print(f"Question (Thai): {question_th}")
    print(f"Question (English): {question_en}")
    print(f"Question Translation Time: {formatted_question_translation_time}")
    print(f"Evidence (Thai): {evidence_th}")
    print(f"Evidence (English): {evidence_en}")
    print(f"Evidence Translation Time: {formatted_evidence_translation_time}")
    print(f"Difficulty: {difficulty}")
    print(f"SQL query: {cleaned_sql}")
    print(f"SQL Generation Time: {formatted_generation_time}")
    print("-----------------------------------------------------------------------------------------------------------\n\n")

# สร้าง predict_dev.json
with open('bird/exp_result/gemma3_output_kg/th/predict_dev_th_j2c2j_tran_error.json', 'w', encoding='utf-8') as f:
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
        print(f"Context: {error['context']}")
        print(f"Error Message: {error['error']}")
        print("--------------------")
else:
    print("No errors occurred during generation.")

# แสดงข้อมูลสรุปจาก log
print("\n=== SQL Generation Time Summary ===")
with open(log_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    total_sql_gen_time = 0
    valid_sql_gen_count = 0
    for row in reader:
        gen_time = float(row['generation_time_seconds'])
        if gen_time >= 0:
            total_sql_gen_time += gen_time
            valid_sql_gen_count += 1
    if valid_sql_gen_count > 0:
        avg_sql_gen_time = total_sql_gen_time / valid_sql_gen_count
        print(f"Average SQL Generation Time: {format_time(avg_sql_gen_time)} (based on {valid_sql_gen_count} successful generations)")
    else:
        print("No successful SQL generations to calculate average time.")

# แสดงข้อมูลสรุปเวลาแปล
print("\n=== Translation Time Summary ===")
with open(log_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    total_question_translation_time = 0
    total_evidence_translation_time = 0
    valid_question_translation_count = 0
    valid_evidence_translation_count = 0
    for row in reader:
        question_trans_time = float(row['question_translation_time'])
        evidence_trans_time = float(row['evidence_translation_time'])
        if question_trans_time >= 0:
            total_question_translation_time += question_trans_time
            valid_question_translation_count += 1
        if evidence_trans_time >= 0:
            total_evidence_translation_time += evidence_trans_time
            valid_evidence_translation_count += 1
    if valid_question_translation_count > 0:
        avg_question_translation_time = total_question_translation_time / valid_question_translation_count
        print(f"Average Question Translation Time: {format_time(avg_question_translation_time)} (based on {valid_question_translation_count} successful translations)")
    else:
        print("No successful question translations to calculate average time.")
    if valid_evidence_translation_count > 0:
        avg_evidence_translation_time = total_evidence_translation_time / valid_evidence_translation_count
        print(f"Average Evidence Translation Time: {format_time(avg_evidence_translation_time)} (based on {valid_evidence_translation_count} successful translations)")
    else:
        print("No successful evidence translations to calculate average time.")