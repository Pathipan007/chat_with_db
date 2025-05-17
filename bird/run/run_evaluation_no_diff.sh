db_root_path='./data/dev/dev_databases/'
predicted_sql_path_kg='./exp_result/gemma3_output_kg/eng/'
predicted_sql_path='./exp_result/gemma3_output/th/'
ground_truth_path='./data/train/train_gold.sql'
num_cpus=16
meta_time_out=60.0
mode_gt='gt'
mode_predict='gpt'

echo '''starting to compare without knowledge for EX'''
python3 -u ./src/evaluation_ex_train_set.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--meta_time_out ${meta_time_out}