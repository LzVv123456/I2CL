import math
import string
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from functools import reduce
import numpy as np
import utils
import global_vars as gv
from peft import get_peft_model, PromptTuningConfig


class ModelWrapper(nn.Module):
    def __init__(self, model, tokenizer, model_config, device):
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.device = device
        self.num_layers = self._get_layer_num()
        self.latent_dict = {}
        self.linear_coef = None
        self.inject_layers = None
        print(f"The model has {self.num_layers} layers:")

    def reset_latent_dict(self):
        self.latent_dict = {}
            
    @contextmanager
    def extract_latent(self):
        handles = []
        try:
            # attach hook
            for layer_idx in range(self.num_layers):
                handles.append(
                    self._get_nested_attr(self._get_arribute_path(layer_idx, 'attn')).register_forward_hook(
                    self.extract_hook_func(layer_idx, 'attn')))
                handles.append(
                    self._get_nested_attr(self._get_arribute_path(layer_idx, 'mlp')).register_forward_hook(
                    self.extract_hook_func(layer_idx, 'mlp')))
                handles.append(
                    self._get_nested_attr(self._get_arribute_path(layer_idx, 'hidden')).register_forward_hook(
                    self.extract_hook_func(layer_idx, 'hidden')))
            yield
        finally:
            # remove hook
            for handle in handles:
                handle.remove()

    def extract_hook_func(self, layer_idx, target_module):
        if layer_idx not in self.latent_dict:
            self.latent_dict[layer_idx] = {}
        def hook_func(module, inputs, outputs):
            if type(outputs) is tuple:
                outputs = outputs[0]
            self.latent_dict[layer_idx][target_module] = outputs.detach().cpu()
        return hook_func
    
    @contextmanager
    def inject_latent(self, context_vector_dict, config, linear_coef, train_mode=False):
        handles = []
        assert self.inject_layers is not None, "inject_layers is not set!"
        inject_method = config['inject_method']
        inject_pos = config['inject_pos']
        add_noise = config['add_noise']
        noise_scale = config['noise_scale']
        try:
            # attach hook
            for layer_idx, layer in enumerate(self.inject_layers):
                for module_idx, module in enumerate(config['module']):
                    context_vector_container = [context_vector_dict[layer][module].to(self.device)]
                    strength = linear_coef[layer_idx, module_idx, :]
                    inject_func = self.inject_hook_func(context_vector_container, strength,
                                                        inject_method, add_noise, noise_scale, 
                                                        inject_pos, train_mode)
                    handles.append(
                        self._get_nested_attr(self._get_arribute_path(layer, module)).
                        register_forward_hook(inject_func)
                        )
            yield
        finally:
            # remove hook
            print(f"Removing {len(handles)} hooks...")
            for handle in handles:
                handle.remove()

    def inject_hook_func(self, context_vector_container, strength, inject_method,
                         add_noise, noise_scale, inject_pos, train_mode=False):

        def hook_func(module, inputs, outputs):
            if type(outputs) is tuple:
                output = outputs[0]     
            else:
                output = outputs
            # init context_vector
            context_vector = context_vector_container[0]
            # expand inject_value to match output size (b, seq_len, d)
            context_vector = context_vector.expand(output.size(0), output.size(1), context_vector.size(-1))
            
            if inject_method == 'add':
                output = output + F.relu(strength) * context_vector
            elif inject_method == 'linear':
                if inject_pos == 'all':
                    output = strength[1] * output + strength[0] * context_vector
                else:
                    if inject_pos == 'last':
                        for i in range(output.size(0)):
                            end_idx = gv.ATTN_MASK_END[i] - 1
                            content = strength[1] * output[i, end_idx, :].clone().detach() + strength[0] * context_vector[i, end_idx, :]
                            output[i, end_idx, :] = content
                    elif inject_pos == 'first':
                        content = strength[1] * output[:, 0, :].clone().detach() + strength[0] * context_vector[:, 0, :]
                        output[:, 0, :] = content
                    elif inject_pos == 'random':
                        for i in range(output.size(0)):
                            end_idx = gv.ATTN_MASK_END[i]
                            random_idx = random.randint(0, end_idx)
                            content = strength[1] * output[i, random_idx, :].clone().detach() + strength[0] * context_vector[i, random_idx, :]
                            output[i, random_idx, :] = content
                    else:
                        raise ValueError("only support all, last, first or random!")
                    
            elif inject_method == 'balance':
                a, b = strength[0], strength[1]
                output = ((1.0 - a) * output + a * context_vector) * b
            else:
                raise ValueError("only support add, linear or balance!")

            if add_noise and train_mode:
                # get l2_norm of output and use it as a scalar to scale noise, make sure no gradient
                output_norm = torch.norm(output, p=2, dim=-1).detach().unsqueeze(-1)
                noise = torch.randn_like(output).detach()
                output += noise * output_norm * noise_scale
            
            if type(outputs) is tuple:
                outputs = list(outputs)
                outputs[0] = output
                outputs = tuple(outputs)
            else:
                outputs = output
            return outputs
        return hook_func
    

    @contextmanager
    def replace_latent(self, context_vector_dict, target_layers, config):
        handles = []
        try:
            # attach hook
            for _, layer in enumerate(target_layers):
                for _, module in enumerate(config['module']):
                    context_vector_container = [context_vector_dict[layer][module].to(self.device)]
                    inject_func = self.replace_hook_func(context_vector_container)
                    handles.append(
                        self._get_nested_attr(self._get_arribute_path(layer, module)).
                        register_forward_hook(inject_func))
            yield
        finally:
            # remove hook
            print(f"Removing {len(handles)} hooks...")
            for handle in handles:
                handle.remove()

    def replace_hook_func(self, context_vector_container):
        def hook_func(module, inputs, outputs):
            if type(outputs) is tuple:
                output = outputs[0]     
            else:
                output = outputs
            # init context_vector
            context_vector = context_vector_container[0]
            # replace hidden states of last token position with context_vector
            for i in range(output.size(0)):
                end_idx = gv.ATTN_MASK_END[i]
                output[i, end_idx, :] = context_vector
            
            if type(outputs) is tuple:
                outputs = list(outputs)
                outputs[0] = output
                outputs = tuple(outputs)
            else:
                outputs = output
            return outputs
        return hook_func
    

    def get_context_vector(self, all_latent_dicts, config):
        if len(all_latent_dicts) == 1:
            latent_dict = all_latent_dicts[0]
            output_dict = {}
            for layer, sub_dict in latent_dict.items():
                output_dict[layer] = {}
                for module in config['module']:
                    latent_value = sub_dict[module]
                    if config['tok_pos'] == 'last':
                        latent_value = latent_value[:, -1, :].squeeze()
                    elif config['tok_pos'] == 'first':
                        latent_value = latent_value[:, 0, :].squeeze()
                    elif config['tok_pos'] == 'random':
                        latent_value = latent_value[:, random.randint(0, latent_value.size(1)), :].squeeze()
                    else:
                        raise ValueError("only support last, first or random!")
                    output_dict[layer][module] = latent_value.detach().to('cpu')
        else:
            # concatenate context vector for each module
            ensemble_dict = {module:[] for module in config['module']} # {module_name: []}
            for _, latent_dict in enumerate(all_latent_dicts):
                cur_dict = {module:[] for module in config['module']}  # {module_name: []}
                for layer, sub_dict in latent_dict.items():
                    for module in config['module']:
                        latent_value = sub_dict[module]  # (b, seq_len, d)  
                        if config['tok_pos'] == 'last':
                            latent_value = latent_value[:, -1, :].squeeze()
                        elif config['tok_pos'] == 'first':
                            latent_value = latent_value[:, 0, :].squeeze()
                        elif config['tok_pos'] == 'random':
                            latent_value = latent_value[:, random.randint(0, latent_value.size(1)), :].squeeze()
                        else:
                            raise ValueError("only support last, first or random!")
                        cur_dict[module].append(latent_value)

                for module, latent_list in cur_dict.items():
                    cur_latent = torch.stack(latent_list, dim=0) # (layer_num, d)
                    ensemble_dict[module].append(cur_latent)

            for module, latent_list in ensemble_dict.items():
                if config['post_fuse_method'] == 'mean':
                    context_vector = torch.stack(latent_list, dim=0).mean(dim=0)  # (layer_num, d)
                    ensemble_dict[module] = context_vector 
                elif config['post_fuse_method'] == 'pca':
                    latents = torch.stack(latent_list, dim=0)  # (ensemble_num, layer_num, d)
                    ensemble_num, layer_num, d = latents.size()
                    latents = latents.view(ensemble_num, -1)  # (ensemble_num*layer_num, d)
                    # apply pca
                    pca = utils.PCA(n_components=1).to(latents.device).fit(latents.float())
                    context_vector = (pca.components_.sum(dim=0, keepdim=True) + pca.mean_).mean(0)
                    ensemble_dict[module] = context_vector.view(layer_num, d)  # (layer_num, d)
                else:
                    raise ValueError("Unsupported ensemble method!")
            # reorganize ensemble_dict into layers
            layers = list(all_latent_dicts[0].keys())
            output_dict = {layer: {} for layer in layers} 
            for module, context_vector in ensemble_dict.items():
                for layer_idx, layer in enumerate(layers):
                    output_dict[layer][module] = context_vector[layer_idx, :].detach().to('cpu')  # (d)

        return output_dict
    

    def calibrate_strength(self, context_vector_dict, dataset, config, 
                           save_dir=None, run_name=None):
        # prepare label dict          
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")

        # frozen all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # init optimizer
        optim_paramters = [{'params': self.linear_coef}]
        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'], 
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'], 
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')

        # get all_data
        all_data = dataset.all_data
        
        # init lr_scheduler
        epochs, batch_size = config['epochs'], config['grad_bs']
        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05*epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
                    if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # train
        print('Calibrating strength params...')
        with self.inject_latent(context_vector_dict, config,
                                self.linear_coef, train_mode=True):
            loss_list = []
            all_data_index = list(range(len(all_data)))
            epoch_iter = len(all_data) // batch_size
            for _ in range(epochs):
                epoch_loss = []
                for i in range(epoch_iter):
                    np.random.shuffle(all_data_index)
                    batch_index = all_data_index[:batch_size]
                    batch_data = [all_data[idx] for idx in batch_index]
                    batch_input, batch_label = [], []
                    for data in batch_data:
                        input_str, ans_list, label = dataset.apply_template(data)

                        # collect single demonstration example
                        if config['cali_example_method'] == 'normal':
                            pass
                        elif config['cali_example_method'] == 'random_label':
                            label = random.choice(list(range(len(ans_list))))
                        else:
                            raise ValueError("only support normal or random_label!")
                        
                        batch_input.append(input_str)
                        batch_label.append(label)

                    input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                    input_ids = input_tok['input_ids'].to(self.device)
                    attn_mask = input_tok['attention_mask'].to(self.device)
                    pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                    # set global vars
                    gv.ATTN_MASK_END = pred_loc
                    gv.ATTN_MASK_START = torch.zeros_like(pred_loc)
                    # forward
                    logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
                    # get prediction logits
                    pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                    # get loss
                    gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                    epoch_loss.append(loss.item())
                    # update strength params
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    cur_lr = optimizer.param_groups[0]['lr']
                    print(f'Epoch {_+1}/{epochs}, batch {i//batch_size+1}/{len(all_data)//batch_size+1}, loss: {loss.item()}, lr: {cur_lr}')
                epoch_loss = np.mean(epoch_loss)
                loss_list.append(epoch_loss)

        # fronzen all learnable strength params
        self.linear_coef.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')


    def softprompt(self, config, dataset, save_dir=None, run_name=None):
        pt_config = PromptTuningConfig(**config['pt_config'])
        peft_model = get_peft_model(self.model, pt_config)

        # prepare label dict          
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")

        # print trainable parameters
        peft_model.print_trainable_parameters()
        print(f'PEFT model:\n {peft_model}')
        # set model to peft model
        self.model = peft_model

        # init optimizer
        optim_paramters = [{'params': self.model.parameters()}]
        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'],
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'], 
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')
        
        # get all data
        all_data = dataset.all_data

        # init lr_scheduler
        epochs, batch_size = config['epochs'], config['grad_bs']
        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05*epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
                    if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # train
        loss_list = []
        all_data_index = list(range(len(all_data)))
        for _ in range(epochs):
            epoch_loss = []
            np.random.shuffle(all_data_index)
            for i in range(0, len(all_data), batch_size):
                batch_index = all_data_index[i: i + batch_size]
                batch_data = [all_data[idx] for idx in batch_index]
                batch_input, batch_label = [], []
                for data in batch_data:
                    input_str, _, label = dataset.apply_template(data)
                    batch_input.append(input_str)
                    batch_label.append(label)

                input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                input_ids = input_tok['input_ids'].to(self.device)
                attn_mask = input_tok['attention_mask'].to(self.device)
                pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                # forward
                logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
                # get prediction logits
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                epoch_loss.append(loss.item())
                # update strength params
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            epoch_loss = np.mean(epoch_loss)
            loss_list.append(epoch_loss)

        # fronzen all learnable strength params
        for param in self.model.parameters():
            param.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')
        

    def init_strength(self, config):
        # get linear_coef size
        if type(config['layer']) == str:
            if config['layer'] == 'all':
                layers = list(range(self.num_layers))
                layer_dim = len(layers)
            elif config['layer'] == 'late':
                layers = list(range((self.num_layers*2)//3, self.num_layers))
                layer_dim = len(layers)
            elif config['layer'] == 'early':
                layers = list(range(self.num_layers//3))
                layer_dim = len(layers)
            elif config['layer'] == 'mid':
                layers = list(range(self.num_layers//3, (self.num_layers*2)//3))
                layer_dim = len(layers)
        elif type(config['layer']) == list:
            layers = config['layer']
            layer_dim = len(layers)
        else:
            raise ValueError("layer must be all, late, early, mid or a list of layer index!")

        if config['inject_method'] == 'add':
            param_size = (layer_dim, len(config['module']), 1)  # (layer_num, module_num, 1)
        elif config['inject_method'] in ['linear', 'balance']:
            param_size = (layer_dim, len(config['module']), 2)  # (layer_num, module_num, 2)
        else:
            raise ValueError("only support add, linear or balance!")
        # set inject_layers
        self.inject_layers = layers
        # init linear_coef
        linear_coef = torch.zeros(param_size, device=self.device) 
        linear_coef += torch.tensor(config['init_value'], device=self.device)
        self.linear_coef = nn.Parameter(linear_coef)
        print(f"linear_coef shape: {self.linear_coef.shape}\n")
        if not self.linear_coef.is_leaf:
            raise ValueError("linear_coef is not a leaf tensor, which is required for optimization.")
        

    def init_noise_context_vector(self, context_vector_dict):
        # init learnable context_vector
        for layer, sub_dict in context_vector_dict.items():
            for module, latent in sub_dict.items():
                noise_vector = torch.randn_like(latent).detach().cpu()
                context_vector_dict[layer][module] = noise_vector
        return context_vector_dict
            
                    
    def _get_nested_attr(self, attr_path):
        """
        Accesses nested attributes of an object based on a dot-separated string path.

        :param obj: The object (e.g., a model).
        :param attr_path: A dot-separated string representing the path to the nested attribute.
                        For example, 'transformer.h' or 'model.layers'.
        :return: The attribute at the specified path.
        """
        try:
            return reduce(getattr, attr_path.split('.'), self.model)
        except AttributeError:
            raise AttributeError(f"Attribute path '{attr_path}' not found.")
        
    def _get_layer_num(self):
        raise NotImplementedError("Please implement get_layer_num function for each model!")
    
    def _get_arribute_path(self, layer_idx, target_module):
        raise NotImplementedError("Please implement get_arribute_path function for each model!")

            
class LlamaWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, model_config, device):
        super().__init__(model, tokenizer, model_config, device)
        self.embed_matrix = self.model.model.embed_tokens.weight.data
        self.embed_dim = self.model_config.hidden_size
        self.last_norm = self.model.model.norm
        
    def _get_layer_num(self):
        return len(self.model.model.layers)
    
    def _get_arribute_path(self, layer_idx, target_module):
        if target_module == "attn":
            return f"model.layers.{layer_idx}.self_attn"
        elif target_module == "mlp":
            return f"model.layers.{layer_idx}.mlp"
        elif target_module == "hidden":
            return f"model.layers.{layer_idx}"
        else:
            raise ValueError("only support att or mlp!")


class GPTWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, model_config, device):
        super().__init__(model, tokenizer, model_config, device)
        self.embed_matrix = self.model.transformer.wte.weight.data
        self.embed_dim = self.embed_matrix.size(-1)
        self.last_norm = self.model.transformer.ln_f
        
    def _get_layer_num(self):
        return len(self.model.transformer.h)
    
    def _get_arribute_path(self, layer_idx, target_module):
        if target_module == "attn":
            return f"transformer.h.{layer_idx}.attn"
        elif target_module == "mlp":
            return f"transformer.h.{layer_idx}.mlp"
        elif target_module == "hidden":
            return f"transformer.h.{layer_idx}"
        else:
            raise ValueError("only support att or mlp!")