import sys
sys.path.append('../')
import my_datasets as md


config = {}
# general
config['exp_name'] = 'exps/i2cl'
config['gpus'] = ['0']
config['models'] = ['meta-llama/Llama-2-7b-hf']  # 'gpt2-xl', 'EleutherAI/gpt-j-6B'
config['datasets'] = list(md.target_datasets.keys())
config['seed'] = 42
config['run_num'] = 5  # number of runs
config['run_baseline'] = True  # whether run baseline
config['metric'] = 'acc'  # 'acc', 'macro_f1'
config['bs'] = 2  # batch size
config['load_in_8bit'] = False
config['use_cache'] = True  # whether use kv cache
config['demo_sample_method'] = 'random' # 'random' or deficient

# calibrate
config['add_noise'] = True  # whether add noise
config['noise_scale'] = 0.001  # noise scale
config['epochs'] = 100  # number of epochs
config['optim'] = 'adamW'  # 'adam', 'adamW', 'sgd'
config['grad_bs'] = 2  # batch size for clibration
config['lr'] = 0.01
config['wd'] = 1e-3
config['cali_example_method'] = 'normal' # 'normal', 'random_label'

# context vector
config['layer'] = 'all' # all, early, mid, late
config['tok_pos'] = 'last'  # 'random', 'first', 'last'
config['inject_method'] = 'linear'  # 'linear', 'constraint', 'add'
config['inject_pos'] = 'all'  # 'all', 'first', last', 'random'
config['init_value'] = [0.1, 1.0]  # linear and constraint: [0.1, 1.0], add: [0.1]
config['module'] = ['mlp', 'attn']  # 'mlp', 'attn', 'hidden'
config['gen_cv_method'] = 'context'  # 'context', 'noise'
config['post_fuse_method'] = 'mean'  # 'mean', 'pca'
config['split_demon'] = True  # split demonstraiton into seperate examples
config['gen_example_method'] = 'normal'  # 'normal', 'random_label', 'no_template', 'random_order'

# data
config['shot_per_class'] = 5  # number of shots per class
config['val_data_num'] = 32 
config['test_data_num'] = 500  # number of test data
config['sample_method'] = 'uniform'  # 'random', 'uniform'
config['use_instruction'] = False
config['add_extra_query'] = False
config['example_separator'] = '\n' 