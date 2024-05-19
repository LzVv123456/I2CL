import sys
sys.path.append('../')
import my_datasets as md


config = {}

config['gpus'] = ['0']
config['exp_name'] = 'exps/i2cl_infer'
config['models'] = ['meta-llama/Llama-2-7b-hf'] 
config['datasets'] = list(md.target_datasets.keys())
config['run_baseline'] = True
config['downstream_datasets'] = None  # None will use the same dataset as source, one can also specify a list of target downstream datasets

config['target_path'] = 'exps/i2cl'
config['use_new_demon'] = True  # whether to use new demonstrations to generate context vectors