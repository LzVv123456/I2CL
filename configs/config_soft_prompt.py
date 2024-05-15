import sys
sys.path.append('../')
import my_datasets as md


config = {}
# general
config['exp_name'] = 'exps/soft_prompt'
config['gpus'] = ['0']
config['models'] = ['meta-llama/Llama-2-7b-hf'] # 'gpt2-xl', 'meta-llama/Llama-2-7b-hf', 'EleutherAI/gpt-j-6B'
config['datasets'] = list(md.target_datasets.keys())
config['seed'] = 42
config['run_num'] = 5
config['run_baseline'] = True
config['metric'] = 'acc'  # 'acc', 'macro_f1'
config['bs'] = 2
config['load_in_8bit'] = False
config['use_cache'] = True
config['example_separator'] = '\n'

# data
config['shot_per_class'] = 5
config['test_data_num'] = 500
config['sample_method'] = 'uniform'  # 'random', 'uniform'
config['add_extra_query'] = False

# prompt_tuning
pt_config = {}
pt_config['task_type'] = 'CAUSAL_LM'
pt_config['num_virtual_tokens'] = 1
pt_config['num_layers'] = 28
config['pt_config'] = pt_config

# optimization
config['epochs'] = 50
config['optim'] = 'adamW'  # 'adam', 'adamW', 'sgd'
config['grad_bs'] = 4
config['lr'] = 0.1
config['wd'] = 1e-3