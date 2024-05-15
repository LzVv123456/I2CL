import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import global_vars as gv
import my_datasets as md


class Evaluator(nn.Module):

    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def evaluate(self, model_wrapper, tokenizer, demonstration='', use_cache=False):
        
        return self._evaluate_text_classification_batch(model_wrapper, tokenizer, 
                                                        demonstration, use_cache=use_cache)

    def _evaluate_text_classification_batch(self, model_wrapper, tokenizer, 
                                            demonstration, use_cache=False):
        
        model = model_wrapper.model
        # prepare label dict          
        label_map = {}
        ans_txt_list = self.dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = tokenizer.encode(ans_txt, add_special_tokens=False)[0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[ans_tok] = label  # index is the label
        print(f"label_map: {label_map}")

        # prepare all data
        all_pred_labels = []
        all_inputs, all_labels = [], []
        for data in self.dataset.all_data:
            ques_str, _, label = self.dataset.apply_template(data)
            if use_cache or len(demonstration) == 0:
                context = ques_str
            else:
                context = demonstration + ques_str
            all_inputs.append(context)
            all_labels.append(label)
            
        # cache the demonstration
        if len(demonstration) > 0 and use_cache:
            demon_token = tokenizer(demonstration, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                demon_outputs = model(**demon_token, use_cache=True)
            demon_past_key_values = demon_outputs.past_key_values
            demon_attn_mask = demon_token['attention_mask']
            demon_past_key_values = tuple(tuple(t.repeat(self.batch_size, 1, 1, 1) for 
                                                t in tup) for tup in demon_past_key_values)
            demon_attn_mask = demon_attn_mask.repeat(self.batch_size, 1)
            if len(all_inputs) % self.batch_size != 0:  # last batch
                sp_demon_past_key_values = tuple(tuple(t.repeat(len(all_inputs) % self.batch_size, 1, 1, 1) 
                                                    for t in tup) for tup in demon_outputs.past_key_values)
                sp_demon_attn_mask = demon_attn_mask[-(len(all_inputs) % self.batch_size):]
            use_cache = True
        else:
            demon_past_key_values = None
            sp_demon_past_key_values = None
            sp_demon_attn_mask = None
            use_cache = False

        # loop over all data
        with torch.no_grad():
            for i in range(0, len(all_inputs), self.batch_size):
                cur_inputs = all_inputs[i:i+self.batch_size]
                # accommodate for the last batch
                if len(cur_inputs) != self.batch_size: 
                    demon_past_key_values = sp_demon_past_key_values
                    demon_attn_mask = sp_demon_attn_mask
                input_tok = tokenizer(cur_inputs, return_tensors="pt", padding=True)
                input_ids = input_tok['input_ids'].to(model.device)
                attn_mask = input_tok['attention_mask'].to(model.device)
                # get index for prediction logits, need to be applied before concatenating demon_attn_mask with attn_mask
                pred_loc = utils.last_one_indices(attn_mask).to(model.device)
                # set global variables
                gv.ATTN_MASK_START = torch.zeros_like(pred_loc)
                gv.ATTN_MASK_END = pred_loc
                if use_cache:
                    attn_mask = torch.cat([demon_attn_mask, attn_mask], dim=1)
                    output = model(input_ids=input_ids, attention_mask=attn_mask,
                                   past_key_values=demon_past_key_values, use_cache=use_cache)
                else:
                    output = model(input_ids=input_ids, attention_mask=attn_mask)
                logits = output.logits

                # get prediction logits
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get prediction labels
                interest_index = list(label_map.keys())
                pred_logits = pred_logits[:, interest_index]
                probs = F.softmax(pred_logits, dim=-1)
                pred_labels = probs.argmax(dim=-1)
                # save results
                all_pred_labels.extend(pred_labels.cpu().numpy().tolist())

        assert len(all_pred_labels) == len(all_labels)
        # both all_results and all_labels are list containing label index, can you help me to calculate accuracy and macro f1?
        # initialize TP, FP, FN
        acc = []
        num_classes = self.dataset.class_num
        TP = [0] * num_classes
        FP = [0] * num_classes
        FN = [0] * num_classes
        for i, true_label in enumerate(all_labels):
            pred_label = all_pred_labels[i]
            pred = pred_label == true_label
            acc.append(pred)
            # Update TP, FP, FN
            if pred:
                TP[true_label] += 1
            else:
                FP[pred_label] += 1
                FN[true_label] += 1
        # Calculate precision, recall, F1 for each class and macro F1
        precision = [0] * num_classes
        recall = [0] * num_classes
        f1 = [0] * num_classes
        for i in range(num_classes):
            precision[i] = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0
            recall[i] = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        macro_f1 = sum(f1) / num_classes
        acc = sum(acc) / len(acc)
        return {'acc': acc, 'macro_f1': macro_f1}