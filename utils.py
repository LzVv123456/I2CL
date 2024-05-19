import os
import sys
import json
import random
import functools
import warnings
import numpy as np
import torch
import torch.nn as nn
from typing import Union, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import matplotlib.pyplot as plt
import wrapper
import my_datasets as md 


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def set_device(gpu_id):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    return device


def init_exp_path(args, exp_name, separate_dataset=True):
    if separate_dataset:
        save_dir = os.path.join(exp_name, args.model_name, args.dataset_name)
    else:
        save_dir = os.path.join(exp_name, args.model_name)
    args.save_dir = save_dir
    if os.path.exists(save_dir) and 'debug' not in exp_name:
        raise ValueError(f"Experiment {exp_name} already exists! please delete it or change the name!")
    os.makedirs(save_dir, exist_ok=True)
    # save config_dict
    with open(f'{save_dir}/config.json', 'w') as f:
        json.dump(args.config, f, indent=4)
    # save args as txt file, I want to make it beautiful
    with open(f'{save_dir}/args.txt', 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')


def load_model_tokenizer(model_name, device, output_hidden_states=True, load_in_8bit=False):
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 output_hidden_states=output_hidden_states, 
                                                 load_in_8bit=load_in_8bit, 
                                                 torch_dtype=torch.float32)
    if not load_in_8bit:
        model = model.to(device)
    config = AutoConfig.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return model, tokenizer, config


def get_model_wrapper(model_name, model, tokenizer, model_config, device):
    if 'llama' in model_name:
        model_wrapper = wrapper.LlamaWrapper(model, tokenizer, model_config, device)
    elif 'gpt' in model_name:
        model_wrapper = wrapper.GPTWrapper(model, tokenizer, model_config, device)
    else:
        raise ValueError("only support llama or gpt!")
    return model_wrapper


def load_config(file_path):
    if not file_path:
        raise ValueError("No file path provided")
    file_dir = os.path.dirname(file_path)
    if file_dir not in sys.path:
        sys.path.append(file_dir)
    file_name = os.path.basename(file_path)
    module_name = os.path.splitext(file_name)[0]
    module = __import__(module_name)
    try:
        my_variable = getattr(module, 'config')
        print(my_variable)
        return my_variable
    except AttributeError:
        print(f"The module does not have a variable named 'config'")


def get_shot_num(dataset, shot_per_class, shot_num=5):
    if hasattr(dataset, 'class_num') and dataset.class_num is not None:
        shot_num = dataset.class_num * shot_per_class
    else:
        shot_num = shot_num
    # if shot_num < 0, then use all data
    if shot_num < 0:
        shot_num = -1
    return shot_num


def first_one_indices(tensor):
    """
    Finds the index of the first 1 in each row of a 2D tensor.

    Args:
      tensor (torch.Tensor): A 2D tensor of size (N, M) containing only 0 and 1 entries.

    Returns:
      torch.Tensor: A tensor of size N containing the index of the first 1 in each row.
                    If a row contains only 0s, the index will be set to -1 (or a sentinel value of your choice).
    """
    # Check for rows containing only zeros.
    is_all_zero = tensor.sum(dim=1) == 0
    # Get the index of the first occurrence of the maximum value (1) along each row.
    indices = tensor.argmax(dim=1)
    # Handle rows with all zeros.
    indices[is_all_zero] = -1  # Set to -1 to indicate no '1' found in these rows
    return indices


def last_one_indices(tensor):
    """
    Finds the index of the last 1 in each row of a 2D tensor.

    Args:
      tensor (torch.Tensor): A 2D tensor of size (N, M) containing only 0 and 1 entries.

    Returns:
      torch.Tensor: A tensor of size N containing the index of the last 1 in each row.
                    If a row contains only 0s, the index will be set to -1 (or a sentinel value of your choice).
    """
    # Reverse each row to find the last occurrence of 1 (which becomes the first in the reversed row)
    reversed_tensor = torch.flip(tensor, [1])
    # Check for rows containing only zeros in the reversed tensor
    is_all_zero = reversed_tensor.sum(dim=1) == 0
    # Get the index of the first occurrence of the maximum value (1) along each row in the reversed tensor
    indices = reversed_tensor.argmax(dim=1) 
    # Adjust the indices for the original order of each row
    indices = tensor.size(1) - 1 - indices
    # Handle rows with all zeros
    indices[is_all_zero] = -1  # Set to -1 to indicate no '1' found in these rows
    return indices


def plot_loss_curve(loss_list, save_path):
    plt.plot(loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path)
    plt.close()


def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_
    

class ContextSolver:
    def __init__(self, task_name, tokenizer=None):
        # assert task_name in ['sst2', 'trec', 'agnews', 'emo']
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.task_dataset = md.get_dataset(task_name, split='train', max_data_num=10)
        self.format_s = self.task_dataset.get_dmonstration_template()['input']
        self.parse_format_s()

    def parse_format_s(self):
        self.X_prefix = self.format_s.split('\n')[0].split(':')[0] + ':'
        self.Y_prefix = self.format_s.split('\n')[1].split(':')[0] + ':'

    def get_empty_demo_context(self, context: str, only_demo_part=True):
        context = context.split('\n')
        for i, line in enumerate(context[:-2]):
            if self.X_prefix in line:
                line = self.X_prefix
            elif self.Y_prefix in line:
                line = line
            else:
                raise warnings.warn('Global prefix or other str exists!')
            context[i] = line
        if only_demo_part:
            context = context[:-2]
        context = '\n'.join(context)
        return context

    def get_mask_strings_and_match_before(self, context, input_ids, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer
        print('debug tokenizer name :', tokenizer.__class__.__name__)
        if 'Llama' in tokenizer.__class__.__name__:
            sap_token = tokenizer.encode('\n', add_special_tokens=False)[1]
            poss = torch.where(input_ids == sap_token)[0]
        else:
            sap_token = tokenizer.encode('\n', add_special_tokens=False)[0]
            poss = torch.where(input_ids == sap_token)[0]
        print('debug sap_token:', sap_token)
        print('debug poss:', poss)
        if len(poss) >= 2:
            match_before = poss[-2] + 1
        else:
            match_before = None

        list_s = []
        list_s.append(self.X_prefix)
        list_s.append('\n' + self.X_prefix)
        context = context.split('\n')
        for i, line in enumerate(context[:-2]):
            if self.X_prefix in line:
                pass
            elif self.Y_prefix in line:
                list_s.append('\n' + line)
                list_s.append('\n' + line + '\n')
            else:
                raise warnings.warn('Global prefix or other str exists!')
        return list_s, match_before

    def get_mask(self, input_ids, tokenizer=None):
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        if len(input_ids.shape) == 2:
            assert input_ids.shape[0] == 1
            input_ids = input_ids[0]
        if tokenizer is None:
            tokenizer = self.tokenizer
        context = tokenizer.decode(input_ids)
        list_s, match_before = self.get_mask_strings_and_match_before(context, input_ids=input_ids,
                                                                      tokenizer=tokenizer)
        print('debug context:', context)
        print('debug list_s:', list_s)
        print('debug match_before:', match_before)
        tensor_str_finder = TensorStrFinder(tokenizer=tokenizer)
        mask = tensor_str_finder.get_strs_mask_in_tensor(list_s=list_s, t=input_ids,
                                                         match_before=match_before)
        return mask
    

class TensorStrFinder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def find_tensor_in_tensor(self, a_tensor: Union[torch.Tensor, list], b_tensor: torch.Tensor,
                              return_mask=True, match_before: Optional[int] = None):
        if len(b_tensor.shape) == 2:
            assert b_tensor.shape[0] == 1
            b_tensor = b_tensor[0]
        if isinstance(a_tensor, list):
            a_tensor = torch.tensor(a_tensor)
        if a_tensor.device != b_tensor.device:
            a_tensor = a_tensor.to(b_tensor.device)

        window_size = len(a_tensor)
        b_windows = b_tensor.unfold(0, window_size, 1)

        matches = torch.all(b_windows == a_tensor, dim=1)

        positions = torch.nonzero(matches, as_tuple=True)[0]

        if return_mask:
            mask = torch.zeros_like(b_tensor, dtype=torch.bool)
            for pos in positions:
                if match_before is None or pos + window_size <= match_before:
                    mask[pos:pos + window_size] = True
            return mask

        return positions

    def find_str_in_tensor(self, s: str, t: torch.Tensor, return_mask=True, match_before=None):
        s_tokens = self.tokenizer.encode(s, add_special_tokens=False)
        s_tensor = torch.LongTensor(s_tokens)
        return self.find_tensor_in_tensor(s_tensor, t, return_mask=return_mask,
                                          match_before=match_before)

    def get_strs_mask_in_tensor(self, list_s: List[str], t: torch.Tensor, match_before=None):
        list_s_tokens = [self.tokenizer.encode(s, add_special_tokens=False) for s in list_s]
        if 'Llama' in self.tokenizer.__class__.__name__:
            list_s_tokens = [s_tokens[1:] if s_tokens[0] == 29871 else s_tokens for s_tokens in list_s_tokens]
        list_s_tensor = [torch.LongTensor(s_tokens) for s_tokens in list_s_tokens]
        print('debug list_s_tensor:', list_s_tensor)
        mask_tensor_list = [
            self.find_tensor_in_tensor(s_tensor, t, return_mask=True, match_before=match_before) for
            s_tensor in list_s_tensor]
        mask_tensor = functools.reduce(torch.logical_or, mask_tensor_list)
        return mask_tensor