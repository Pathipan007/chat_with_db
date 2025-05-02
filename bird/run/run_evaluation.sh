db_root_path='./data/dev/dev_databases/'
data_mode='dev'
diff_json_path='./data/dev/dev_j2c2j.json'
predicted_sql_path_kg='./exp_result/gemma3_output_kg/th/'
predicted_sql_path='./exp_result/gemma3_output/th/'
ground_truth_path='./data/dev/'
num_cpus=16
meta_time_out=60.0
mode_gt='gt'
mode_predict='gpt'

: <<'END_COMMENT'
echo '''starting to compare without knowledge for ex'''
python3 -u ./src/evaluation_ex.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}
END_COMMENT

echo '''starting to compare with knowledge for EX'''
python3 -u ./src/evaluation_ex_log.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}
echo '''Finished evaluation using knowledge(Evidence) for EX'''

: <<'END_COMMENT'
echo '''starting to compare without knowledge for ves'''
python3 -u ./src/evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}

echo '''starting to compare with knowledge for ves'''
python3 -u ./src/evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}
END_COMMENT