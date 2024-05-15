import gc
import os
import json
import time
import copy
import random
import argparse
import itertools
import torch
import numpy as np
from multiprocessing import Process, Queue
from itertools import combinations

import utils
import my_datasets as md
import evaluator as ev


def main(args):
    # load config from target_path
    tar_exp_path = os.path.join(args.config['target_path'],
                                args.model_name, args.dataset_name)
    tar_config_path = os.path.join(tar_exp_path, 'config.json')
    tar_result_path = os.path.join(tar_exp_path, 'result_dict.json')
    # reutrn if target_path does not exist
    if not os.path.exists(tar_config_path) or not os.path.exists(tar_result_path):
        print(f"target_path: {tar_exp_path} does not exist!")
        return
    # load config
    with open(tar_config_path, 'r') as f:
        config = json.load(f)
    # load result_dict
    with open(tar_result_path, 'r') as f:
        result_dict = json.load(f)
    # update config with args.config
    config.update(args.config)
    args.config = config
    args.result_dict = result_dict

    # set global seed
    utils.set_seed(args.config['seed'])
    # set device
    args.device = utils.set_device(args.gpu)
    # set metric used
    args.metric = args.config['metric']
    # get save dir
    utils.init_exp_path(args, args.config['exp_name'])

    # load tokenizer and model
    model, tokenizer, model_config = \
    utils.load_model_tokenizer(args.model_name, args.device)
    
    # get model_wrapper
    model_wrapper = utils.get_model_wrapper(args.model_name, model, 
                                            tokenizer, model_config, 
                                            args.device)
    # load datasets
    train_dataset = md.get_dataset(args.dataset_name, split='train',
                                   max_data_num=None, seed = args.config['seed'])
    holdout_dataset = md.get_dataset(args.dataset_name, split='validation', 
                                     max_data_num=args.config['val_data_num'],
                                     sample_mode=args.config['sample_method'], 
                                     seed=args.config['seed'])
    test_dataset = md.get_dataset(args.dataset_name, split='test', 
                                  max_data_num=args.config['test_data_num'],
                                  sample_mode=args.config['sample_method'], 
                                  seed=args.config['seed'])

    # get shot_num
    if args.dataset_name  == 'dbpedia':  # always use 1-shot for dbpedia
        args.config['shot_per_class'] = 1
        args.config['bs'] = 1
    args.shot_num = utils.get_shot_num(train_dataset, args.config['shot_per_class'])
    # build evaluate
    holdout_evaluator = ev.Evaluator(holdout_dataset, batch_size=args.config['bs'])
    test_evaluator = ev.Evaluator(test_dataset, batch_size=args.config['bs'])
    # init result_dict
    infer_result_dict = {'demon': {},
                         'split_demon': {},
                         'test_result': {'zero_shot': [], 'few_shot': [], 'ours': {}}, 
                         'linear_coef': {},
                         'time': {'calibrate': [], 'evaluate': []}
                         }
    
    
    all_context_vector_dicts, all_linear_coefs = [], []
    for run_id in range(args.config['run_num']):
        run_name = f'run_{run_id}'
        args.run_name = run_name
        print(f'Run time {run_name}')
        run_seed = args.config['seed'] + run_id
        utils.set_seed(run_seed)

        # zero-shot baseline
        if run_id == 0 and args.config['run_baseline']:  
            test_zeroshot_result = test_evaluator.evaluate(model_wrapper, tokenizer, demonstration='',
                                                           use_cache=args.config['use_cache'])
            infer_result_dict['test_result']['zero_shot'].append(test_zeroshot_result)
            print(f'Test zero-shot result: {test_zeroshot_result}\n')
        
        if args.config['use_new_demon']:
            # sample demonstration
            count = 0
            temp_demon_list, temp_result_list = [], []
            while True:
                demon, split_demon, demon_data_index = \
                train_dataset.gen_few_shot_demonstration(tokenizer=tokenizer, shot_num=args.shot_num, 
                                                        max_demonstration_tok_len=1e8,
                                                        add_extra_query=args.config['add_extra_query'],
                                                        example_separator=args.config['example_separator'],
                                                        return_data_index=True, seed=random.randint(0, 1e6) + run_seed)
                temp_demon_list.append((demon, split_demon, demon_data_index))
                    
                if args.config['demo_sample_method'] == 'random':
                    break
                else:
                    tem_val_result = holdout_evaluator.evaluate(model_wrapper, tokenizer, 
                                                                demonstration=demon,
                                                                use_cache=args.config['use_cache'])
                    temp_result = tem_val_result[args.metric]
                    temp_result_list.append(temp_result)
                if count > 20:
                    if args.config['demo_sample_method'] == 'deficient':
                        demon, split_demon, demon_data_index = temp_demon_list[np.argmin(temp_result_list)]
                    else:
                        raise ValueError('Invalid demon_sample_method')
                    break
                count += 1
            
            if args.config['add_extra_query']:
                first_format_anchor = train_dataset.get_dmonstration_template()['format'][0]
                # remove all contents after the last first_format_anchor including the anchor
                if first_format_anchor in demon:
                    baseline_demon = demon[:demon.rfind(first_format_anchor)]
                    query_demon = demon[demon.rfind(first_format_anchor):]
            else:
                baseline_demon = demon
                query_demon = None
            print(f'Demonstration:\n{demon}\n')
            print(f'Baseline demonstration:\n{baseline_demon}\n')
            print(f'Query demonstration:\n{query_demon}\n')
            demon_list = [demon]
            split_demon_list = split_demon
        else:
            demon_list = args.result_dict['demon'][run_name]
            demon = demon_list[0]
            baseline_demon = demon
            try:
                split_demon_list = args.result_dict['split_demon'][run_name]
            except KeyError:
                raise ValueError('split_demon_list not found in result_dict!')
            
        # save demon_list
        infer_result_dict['demon'][run_name] = demon_list
        infer_result_dict['split_demon'][run_name] = split_demon_list

        # few-shot baseline
        if args.config['run_baseline']:
            test_fewshot_result = test_evaluator.evaluate(model_wrapper, tokenizer, 
                                                        demonstration=baseline_demon, 
                                                        use_cache=args.config['use_cache'])
            infer_result_dict['test_result']['few_shot'].append(test_fewshot_result)
            print(f'Test few-shot result: {test_fewshot_result}\n')

        # extract latents ======================================================================
        all_latent_dicts = []
        with torch.no_grad():
            if not args.config['split_demon']:
                target_demon_list = demon_list[0]
            else:
                target_demon_list = split_demon_list
            for cur_demon in target_demon_list:
                with model_wrapper.extract_latent():
                    demon_token = tokenizer(cur_demon, return_tensors='pt').to(args.device)
                    _ = model(**demon_token)
                all_latent_dicts.append(model_wrapper.latent_dict)
                model_wrapper.reset_latent_dict()

        # generate context vector ==============================================================
        context_vector_dict = model_wrapper.get_context_vector(all_latent_dicts, args.config)
        del all_latent_dicts
    
        # get strength params ===================================================================
        model_wrapper.init_strength(args.config)
        del model_wrapper.linear_coef
        model_wrapper.linear_coef = torch.tensor(args.result_dict['linear_coef'][run_name])

        # save context_vector_dict and linear_coef for ensemble
        all_context_vector_dicts.append(context_vector_dict)
        all_linear_coefs.append(model_wrapper.linear_coef)

        # prepare downstream tasks
        if args.config['downstream_datasets'] is None:
            downstream_datasets = [args.dataset_name]
        else:
            downstream_datasets = args.config['downstream_datasets']

        for target_task_name in downstream_datasets:
            # init saving structure
            if target_task_name not in infer_result_dict['test_result']['ours']:
                infer_result_dict['test_result']['ours'][target_task_name] = []

            # prepare target dataset
            target_dataset = md.get_dataset(target_task_name, split='test', 
                                max_data_num=args.config['test_data_num'],
                                sample_mode=args.config['sample_method'], 
                                seed=args.config['seed'])
            target_evaluator = ev.Evaluator(target_dataset, batch_size=args.config['bs'])

            # evaluate i2cl ========================================================================
            s_t = time.time()
            with torch.no_grad():
                with model_wrapper.inject_latent(context_vector_dict, args.config, 
                                                 model_wrapper.linear_coef):
                    test_ours_result = target_evaluator.evaluate(model_wrapper, tokenizer, demonstration='',
                                                                use_cache=args.config['use_cache'])
                    print(f'Test I2CL result: {test_ours_result}\n')
                    infer_result_dict['test_result']['ours'][target_task_name].append(test_ours_result)
            e_t = time.time()

            print(f'Evaluate time: {e_t - s_t}')
            infer_result_dict['time']['evaluate'].append(e_t - s_t)

        # save result_dict after each run
        with open(args.save_dir + '/result_dict.json', 'w') as f:
            json.dump(infer_result_dict, f, indent=4)
                            
    # delete all variables
    del model, tokenizer, model_config, model_wrapper, train_dataset, test_dataset, holdout_dataset
    del test_evaluator, holdout_evaluator
    del all_context_vector_dicts, all_linear_coefs
    del infer_result_dict


# get args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_i2cl_infer.py', help='path to config file')
    return parser.parse_args()


if __name__ == "__main__":
    # get args
    args = get_args()
    # load config
    config = utils.load_config(args.config_path)
    # Generate all combinations of models and datasets
    combinations = list(itertools.product(config['models'], config['datasets']))
    # Queue to hold tasks
    task_queue = Queue()
    for combine in combinations:
        task_queue.put(combine)

    def run_task(gpu_id, config):
        while not task_queue.empty():
            model_name, dataset_name = task_queue.get()
            print(f"Running {model_name} on {dataset_name} with GPU {gpu_id}")
            input_args = argparse.Namespace()
            cur_config = copy.deepcopy(config)
            input_args.model_name = model_name
            input_args.dataset_name = dataset_name
            input_args.gpu = gpu_id
            input_args.config = cur_config
            try:
                main(input_args)
            finally:
                # Clean up CUDA memory after each task
                gc.collect()
                torch.cuda.empty_cache()
                print(f"CUDA memory cleared for GPU {gpu_id}") 
                time.sleep(5)

    # Create a process for each GPU
    processes = [Process(target=run_task, args=(gpu_id, config)) for gpu_id in config['gpus']]
    # Start all processes
    for p in processes:
        p.start()
    # Wait for all processes to finish
    for p in processes:
        p.join()
    print("All tasks completed.")