import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
import csv
import os
from func_timeout import func_timeout, FunctionTimedOut
import pandas as pd
from openpyxl.styles import Alignment
import openpyxl

def load_json(dir):
    with open(dir, 'r') as j:
        contents = json.loads(j.read())
    return contents

def result_callback(result):
    exec_result.append(result)


def execute_sql(predicted_sql,ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res



def execute_model(predicted_sql,ground_truth, db_place, idx, meta_time_out):
    try:
        res = func_timeout(meta_time_out, execute_sql,
                                  args=(predicted_sql, ground_truth, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f'timeout',)]
        res = 0
    except Exception as e:
        result = [(f'error',)]  # possibly len(query) > 512 or not executable
        res = 0
    # print(result)
    # result = str(set([ret[0] for ret in result]))
    result = {'sql_idx': idx, 'res': res}
    # print(result)
    return result


def package_sqls(sql_path, db_root_path, mode='gpt', data_mode='dev'):
    clean_sqls = []
    db_path_list = []
    if mode == 'gpt':
        sql_data = json.load(open(sql_path + 'predict_' + data_mode + '.json', 'r'))
        for idx, sql_str in sql_data.items():
            if type(sql_str) == str:
                sql, db_name = sql_str.split('\t----- bird -----\t')
            else:
                sql, db_name = " ", "financial"
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    elif mode == 'gt':
        sqls = open(sql_path + data_mode + '_gold.sql')
        sql_txt = sqls.readlines()
        # sql_txt = [sql.split('\t')[0] for sql in sql_txt]
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split('\t')
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    return clean_sqls, db_path_list

def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i,sql_pair in enumerate(sqls):

        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out), callback=result_callback)
    pool.close()
    pool.join()

def sort_results(list_of_dicts):
  return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_acc_by_diff(exec_results,diff_json_path):
    num_queries = len(exec_results)
    results = [res['res'] for res in exec_results]
    contents = load_json(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []

    for i,content in enumerate(contents):
        if content['difficulty'] == 'simple':
            simple_results.append(exec_results[i])

        if content['difficulty'] == 'moderate':
            moderate_results.append(exec_results[i])

        if content['difficulty'] == 'challenging':
            challenging_results.append(exec_results[i])

    simple_acc = sum([res['res'] for res in simple_results])/len(simple_results)
    moderate_acc = sum([res['res'] for res in moderate_results])/len(moderate_results)
    challenging_acc = sum([res['res'] for res in challenging_results])/len(challenging_results)
    all_acc = sum(results)/num_queries
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists



def print_data(score_lists,count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))

    print('======================================    ACCURACY    =====================================')
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy', *score_lists))

def run_query_safe(db_path, query):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        return [(f'error: {e}',)]

def safe_execute_query(db_path, query, timeout=10):
    try:
        return func_timeout(timeout, run_query_safe, args=(db_path, query))
    except FunctionTimedOut:
        return [('timeout',)]

def truncate_result(res, max_len=3):
    if isinstance(res, list) and len(res) > max_len:
        res = res[:max_len] + [('... truncated',)]
    return '\n'.join([str(row) for row in res])

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--predicted_sql_path', type=str, required=True, default='')
    args_parser.add_argument('--ground_truth_path', type=str, required=True, default='')
    args_parser.add_argument('--data_mode', type=str, required=True, default='dev')
    args_parser.add_argument('--db_root_path', type=str, required=True, default='')
    args_parser.add_argument('--num_cpus', type=int, default=1)
    args_parser.add_argument('--meta_time_out', type=float, default=30.0)
    args_parser.add_argument('--mode_gt', type=str, default='gt')
    args_parser.add_argument('--mode_predict', type=str, default='gpt')
    args_parser.add_argument('--difficulty',type=str,default='simple')
    args_parser.add_argument('--diff_json_path',type=str,default='')
    args = args_parser.parse_args()
    exec_result = []

    pred_queries, db_paths = package_sqls(args.predicted_sql_path, args.db_root_path, mode=args.mode_predict,
                                          data_mode=args.data_mode)
    # generate gt sqls:
    gt_queries, db_paths_gt = package_sqls(args.ground_truth_path, args.db_root_path, mode='gt',
                                           data_mode=args.data_mode)

    query_pairs = list(zip(pred_queries,gt_queries))
    run_sqls_parallel(query_pairs, db_places=db_paths, num_cpus=args.num_cpus, meta_time_out=args.meta_time_out)
    exec_result = sort_results(exec_result)
   
    # โหลดไฟล์ JSON และ mapping ค่า
    difficulty_contents = load_json(args.diff_json_path)
    id_to_diff = {item['question_id']: item['difficulty'] for item in difficulty_contents}
    id_to_data = {
        item['question_id']: {
            'question': item.get('question', ''),
            'evidence': item.get('evidence', ''),
            'question_th': item.get('question_th', ''),
            'evidence_th': item.get('evidence_th', '')
        }
        for item in difficulty_contents
    }

    # เตรียมไฟล์ .xlsx
    log_path = 'eval_log/th_12b_log.xlsx'
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Evaluation Log'

    # เขียน header
    headers = ['Question ID', 'Difficulty', 'Question (Original)', 'Evidence (Original)',
            'Question (TH)', 'Evidence (TH)', 'Predicted SQL', 'Ground Truth SQL',
            'Predicted Result', 'Ground Truth Result', 'Is Correct']
    ws.append(headers)

    # บันทึก log
    for result in exec_result:
        idx = result['sql_idx']
        print(f"Logging index: {idx+1}/{len(difficulty_contents)}")

        pred_sql, gt_sql = query_pairs[idx]
        db_path = db_paths[idx]

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            predicted_res = safe_execute_query(db_path, pred_sql, timeout=10)
        except Exception as e:
            predicted_res = [(f'error: {e}',)]

        try:
            ground_truth_res = safe_execute_query(db_path, gt_sql, timeout=10)
        except Exception as e:
            ground_truth_res = [(f'error: {e}',)]

        conn.close()

        difficulty = id_to_diff.get(idx, 'unknown')
        is_correct = set(predicted_res) == set(ground_truth_res)
        
        data = id_to_data.get(idx, {})
        question = data.get('question', '')
        evidence = data.get('evidence', '')
        question_th = data.get('question_th', '')
        evidence_th = data.get('evidence_th', '')

        row_data = [
            idx,
            difficulty,
            question,
            evidence,
            question_th,
            evidence_th,
            pred_sql,
            gt_sql,
            truncate_result(predicted_res),
            truncate_result(ground_truth_res),
            is_correct
        ]
        ws.append(row_data)

    # ปรับการแสดงผลให้ wrap ข้อความยาวๆ
    for col in ws.columns:
        for cell in col:
            cell.alignment = Alignment(wrap_text=True)

    # บันทึกไฟล์ .xlsx
    wb.save(log_path)
    print(f"Log saved to {log_path}")

    print('==== start calculate ====')
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = \
        compute_acc_by_diff(exec_result,args.diff_json_path)
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    print_data(score_lists,count_lists)
    print('===========================================================================================')
