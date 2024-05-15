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
                                   max_data_num=None, seed=args.config['seed'])
    test_dataset = md.get_dataset(args.dataset_name, split='test', 
                                  max_data_num=args.config['test_data_num'],
                                  sample_mode=args.config['sample_method'], 
                                  seed=args.config['seed'])

    # get max demonstration token length for each dataset
    if args.config['split_demon']:
        # when split demon, do not check max example token length
        args.test_max_token = 1e8
    else:
        args.test_max_token = test_dataset.get_max_demonstration_token_length(tokenizer)

    # get shot_num
    if args.dataset_name  == 'dbpedia':  # always use 1-shot for dbpedia
        args.config['shot_per_class'] = 1
        args.config['bs'] = 1
    args.shot_num = utils.get_shot_num(train_dataset, args.config['shot_per_class'])
    # build evaluate
    test_evaluator = ev.Evaluator(test_dataset, batch_size=args.config['bs'])
    # init result_dict
    infer_result_dict = {'demon': {},
                         'split_demon': {},
                         'test_result': {'zero_shot': [], 'few_shot': [], 'ours': [], 'ensemble': {}}, 
                         'linear_coef': {},
                         'time': {'calibrate': [], 'evaluate': []}
                         }

    all_cv_dicts, all_coef = [], []
    for dataset_name in list(md.target_datasets.keys()):
        # collect calibrated coefficients
        coe_save_path = os.path.join(args.config['target_path'], args.model_name, dataset_name, 'result_dict.json')
        with open(coe_save_path, 'r') as f:
            cur_result_dict = json.load(f)
        tem_coef = []
        for run_id, coef in cur_result_dict['linear_coef'].items():
            tem_coef.append(torch.tensor(coef))
        # average strength params
        all_coef.append(torch.stack(tem_coef).mean(dim=0))
        
        # collect context vectors
        cv_save_path = os.path.join(args.config['target_path'], args.model_name, dataset_name, 'cv_save_dict.json')
        with open(cv_save_path, 'r') as f:
            cv_dict = json.load(f)
        cur_cv_dict = {}
        for _, cv_dict in cv_dict.items():
            for layer, sub_dict in cv_dict.items():
                if layer not in cur_cv_dict:
                    cur_cv_dict[layer] = {}
                for module, activation in sub_dict.items():
                    if module not in cur_cv_dict[layer]:
                        cur_cv_dict[layer][module] = []
                    cur_cv_dict[layer][module].append(torch.tensor(activation))
        # average context vector diict
        for layer, sub_dict in cur_cv_dict.items():
            for module, activation_list in sub_dict.items():
                cur_cv_dict[layer][module] = torch.stack(activation_list).mean(dim=0)
        all_cv_dicts.append(cur_cv_dict)

    for run_id in range(args.config['run_num']):
        run_name = f'run_{run_id}'
        args.run_name = run_name
        print(f'Run time {run_name}')
        run_seed = args.config['seed'] + run_id
        utils.set_seed(run_seed)

        # build val dataset
        _, split_demon_list, demon_data_index = \
        train_dataset.gen_few_shot_demonstration(tokenizer=tokenizer, shot_num=args.shot_num, 
                                                    max_demonstration_tok_len=args.test_max_token,
                                                    add_extra_query=args.config['add_extra_query'],
                                                    example_separator=args.config['example_separator'],
                                                    gen_example_method = args.config['gen_example_method'],
                                                    return_data_index=True, seed=random.randint(0, 1e6))
        # build val_evaluator use demon_data_index
        val_dataset = copy.deepcopy(train_dataset)
        val_dataset.all_data = [train_dataset.all_data[i] for i in demon_data_index]

        # get demon
        demon_list = args.result_dict['demon'][run_name]
        assert split_demon_list == args.result_dict['split_demon'][run_name], \
        print(f'split_demon_list: {split_demon_list} != {args.result_dict["split_demon"][run_name]}')
    
        # save demon_list
        infer_result_dict['demon'][run_name] = demon_list
        infer_result_dict['split_demon'][run_name] = split_demon_list

        # cal task simlarity ===================================================================
        cur_coef = torch.tensor(args.result_dict['linear_coef'][run_name]).view(-1)
        ref_coef = torch.stack(all_coef)
        ref_coef = ref_coef.view(ref_coef.size(0), -1)
        # calculate cosine similarity between cur_coef and all_coef
        sim = torch.nn.functional.cosine_similarity(cur_coef, ref_coef, dim=1)

        # keep sim and its index whose similarity is larger than threshold
        tar_sim = sim[sim > args.config['threshold']]
        tar_idx = torch.nonzero(sim > args.config['threshold']).view(-1)

        print(f'Similarities: {tar_sim}')
        print(f'Similar tasks: {[list(md.target_datasets.keys())[idx] for idx in tar_idx]}')

        tar_sim = tar_sim.cpu().numpy()
        tar_cv_dicts = [all_cv_dicts[idx] for idx in tar_idx]
        tar_coef = [all_coef[idx] for idx in tar_idx]
        assert len(tar_cv_dicts) > 2, print('No enough transferable tasks!')
        infer_result_dict['similar_task_num'] = len(tar_cv_dicts)

        # change top_k_sim to probability distribution that sums to 1
        def softmax_with_temperature(logits, temperature=1.0):
            scaled_logits = logits / temperature
            exps = np.exp(scaled_logits - np.max(scaled_logits))  # For numerical stability
            softmax_outputs = exps / np.sum(exps)
            return softmax_outputs

        tar_prob = softmax_with_temperature(tar_sim, args.config['temp'])
        context_vector_dict, linear_coef = prepare_inject_dicts_params(tar_prob, tar_cv_dicts, tar_coef)

        # set strength params
        model_wrapper.init_strength(args.config)

        # calibrate context vector 
        s_t = time.time()
        model_wrapper.calibrate_strength(context_vector_dict, val_dataset, 
                                         args.config, save_dir=args.save_dir, 
                                         run_name=args.run_name)
        e_t = time.time()
        print(f'Calibration time: {e_t - s_t}')
        infer_result_dict['time']['calibrate'].append(e_t - s_t)

        # save linear_coef
        infer_result_dict['linear_coef'][run_name] = model_wrapper.linear_coef.tolist()

        # evaluate i2cl ========================================================================
        s_t = time.time()
        with torch.no_grad():
            with model_wrapper.inject_latent(context_vector_dict, args.config, 
                                            model_wrapper.linear_coef):
                test_ours_result = test_evaluator.evaluate(model_wrapper, tokenizer, demonstration='', 
                                                           use_cache=args.config['use_cache'])
                print(f'Test I2CL result: {test_ours_result}\n')
                infer_result_dict['test_result']['ours'].append(test_ours_result)
        e_t = time.time()

        print(f'Evaluate time: {e_t - s_t}')
        infer_result_dict['time']['evaluate'].append(e_t - s_t)

        # save result_dict after each run
        with open(args.save_dir + '/result_dict.json', 'w') as f:
            json.dump(infer_result_dict, f, indent=4)

    # delete all variables
    del model, tokenizer, model_config, model_wrapper, train_dataset, test_dataset, test_evaluator
    del all_cv_dicts, all_coef
    del context_vector_dict, linear_coef
    del infer_result_dict
                             

def prepare_inject_dicts_params(tar_prob, tar_cv_dicts, tar_coef):
    target_layers = list(tar_cv_dicts[0].keys())
    target_modules = list(tar_cv_dicts[0][target_layers[0]].keys())
    print(f'target_layers: {target_layers}')
    print(f'target_modules: {target_modules}')
    # init an empty ensemble_dict with the same structure as all_inject_dicts
    ensemble_cv_dict = {layer: {module: 0 for module in target_modules} for layer in target_layers}
    new_coef = torch.zeros(tar_coef[0].size())
    for idx, cv_dict in enumerate(tar_cv_dicts):
        for layer_idx, layer in enumerate(target_layers):
            for module_idx, module in enumerate(target_modules):
                cv = cv_dict[layer][module]
                coef = tar_coef[idx][layer_idx, module_idx, 0]
                ensemble_cv_dict[layer][module] += cv * coef * tar_prob[idx]
        new_coef += tar_coef[idx] * tar_prob[idx]
    # set the first strength param to 1 since coefficient of context vector has been included in the context vector
    new_coef[:, :, 0] = 1
    # set layer name in ensemble_cv_dict to int type
    ensemble_cv_dict = {int(layer): sub_dict for layer, sub_dict in ensemble_cv_dict.items()}
    return ensemble_cv_dict, new_coef


# get args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_i2cl_transfer_learning.py', help='path to config file')
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