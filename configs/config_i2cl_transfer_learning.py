import sys
sys.path.append('../')
import my_datasets as md

config = {}
config['gpus'] = ['0']
config['exp_name'] = 'exps/i2cl_transfer_learning'
config['models'] = ['meta-llama/Llama-2-7b-hf'] 
config['datasets'] = list(md.target_datasets.keys())

config['run_num'] = 1

config['threshold'] = 0.8
config['temp'] = 0.5
config['target_path'] = 'exps/i2cl'
