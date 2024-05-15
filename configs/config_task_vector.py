import sys
sys.path.append('../')
import my_datasets as md


config = {}
# general
config['exp_name'] = 'exps/task_vector'
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

# context vector
config['layer'] = 'all' # all, late, early, mid
config['tok_pos'] = 'last'
config['module'] = ['hidden']  # 'mlp', 'attn', 'hidden'
config['gen_cv_method'] = 'context'  # 'context', 'noise'
config['post_fuse_method'] = 'mean'  # 'mean', 'pca'
config['split_demon'] = False  # split demonstraiton into seperate examples

# data
config['shot_per_class'] = 5
config['val_data_num'] = 32
config['test_data_num'] = 500
config['sample_method'] = 'uniform'  # 'random', 'uniform'
config['use_instruction'] = False
config['add_extra_query'] = True
config['example_separator'] = '\n'