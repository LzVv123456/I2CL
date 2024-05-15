import random
import string
from itertools import zip_longest
from torch.utils.data import Dataset


class BaseTask(Dataset):
    def __init__(self, task_name, sample_mode='random', max_data_num=None, use_instruction=False, seed=0):
        super().__init__()
        self.task_name = task_name
        self.sample_mode = sample_mode
        self.max_data_num = max_data_num  # maximum number of loaded data
        self.use_instruction = use_instruction  # whether add task instruction as prefix
        self.task_type = None
        self.dataset = None  # dataset
        self.all_data = None  # all data
        self.all_labels = None # all labels
        self.seed = seed

    def random_sample_data(self, max_data_num, seed=0):
        """
        This function is used to randomly sample data from the dataset.
        """
        # set random seed
        random.seed(self.seed)
        assert self.all_data is not None, "Please load data first!"
        if max_data_num < len(self.all_data):
            if (self.all_labels is None) or (self.sample_mode == 'random'):
                # random sample data
                self.all_data = random.sample(self.all_data, max_data_num)
            else:
                # sample data uniformly from each class, if not possible, sample up to max_data_num
                label_dict = {label: [] for label in list(set(self.all_labels))}
                for index, label in enumerate(self.all_labels):
                    label_dict[label].append(self.all_data[index])
                # print key and number of data with current key
                print("Number of data in each class:")
                for label, label_list in label_dict.items():
                    print(f"{label}: {len(label_list)}")
                new_all_data = []

                key_num = len(label_dict.keys())
                for label in label_dict:
                    # if number of data with current label is smaller than max_data_num / class_num, use all data and distibute remaining quota evenly to rest classes
                    if len(label_dict[label]) < max_data_num // key_num:
                        new_all_data.extend(label_dict[label])
                    else:
                        # random sample data from current label
                        new_all_data.extend(random.sample(label_dict[label], max_data_num // key_num))
                # if length of new_all_data is smaller than max_data_num, randomly sample data from all data but check if the data is already in new_all_data
                while len(new_all_data) < max_data_num:
                    tem_data = random.choice(self.all_data)
                    if tem_data not in new_all_data:
                        new_all_data.append(tem_data)
                self.all_data = new_all_data
        else:
            print(f"Warning: max_data_num {max_data_num} is larger than the dataset size {len(self.all_data)}!, use all data instead.")
  

    def get_max_demonstration_token_length(self, tokenizer):
        """
        This function is used to get the maximum token length of the example in the dataset. 
        This is mainly used for test dataset to determine the maximum number of the demonstration 
        tokens that can be used due to the limit of context window.
        """
        all_data_toks = []
        for data in self.all_data:
            input_str, ans_str, _ = self.apply_template(data)
            if self.use_instruction:
                instruct = self.get_task_instruction()
                instruct = instruct + '\n'
            else:
                instruct = ""
            demonstration_str = [(instruct + input_str + ' ' + ans_str[i]) for i in range(len(ans_str))]
            all_data_toks.extend(demonstration_str)
        # get maximum token length of the example in the dataset
        encoded_inputs = tokenizer.batch_encode_plus(all_data_toks, return_tensors="pt", padding=True, truncation=True)
        single_data_max_len = encoded_inputs['attention_mask'].sum(dim=1).max().item()
        if 'Llama' in tokenizer.name_or_path:
            if 'Llama-2' in tokenizer.name_or_path:
                cxt_max_len = 4096
            else:
                cxt_max_len = 2048
        else:
            cxt_max_len = tokenizer.model_max_length
        max_demonstration_len = cxt_max_len - single_data_max_len
        print(f"Max demonstration token length is : {cxt_max_len} - {single_data_max_len} = {max_demonstration_len}",)
        return max_demonstration_len


    def gen_few_shot_demonstration(self, tokenizer, shot_num, max_demonstration_tok_len=1e6,
                                   add_extra_query=False, example_separator='\n', 
                                   return_data_index=False, gen_example_method='normal', seed=0):
                                    
        """
        This function is used to generate few-shot demonstration.
        """
        # set random seed
        random.seed(seed)

        assert self.all_data is not None, "Please load data first!"
        assert shot_num <= len(self.all_data), "Shot number should be smaller than the number of data!"
        if hasattr(self, 'class_num') and self.class_num is not None:  # if class number is provided
            assert shot_num == -1 or shot_num == 0 or shot_num >= self.class_num, "Shot number should be at least larger than the number of classes!"
            class_num = self.class_num
            # get label dict
            label_dict = {label: [] for label in range(class_num)}
            for index, data in enumerate(self.all_data):
                label_dict[self.apply_template(data)[-1]].append(index)
        else:
            class_num = None

        # get task instruction
        instruct = self.get_task_instruction() if self.use_instruction else ""
        if len(instruct) > 0:
            demonstration_expample_list = [instruct]
            demonstration  = instruct + example_separator
        else:
            demonstration_expample_list = []
            demonstration = ""

        if class_num is None:  # random sample data
            sample_indexes = random.sample(range(len(self.all_data)), shot_num)
        else: # uniform sample data from each class
            sample_indexes = []
            # split sample number into each class equally, if not possible, sample as many as possible
            for label in label_dict:
                sample_indexes.extend(random.sample(label_dict[label], shot_num // class_num))
            # random shuffle the sample indexes
            random.shuffle(sample_indexes)  

        for index in sample_indexes:
            input_str, ans_str, label = self.apply_template(self.all_data[index])
            ans = ans_str[label]
            new_example = input_str + ' ' + ans
            demonstration = demonstration + new_example + example_separator
            # collect single demonstration example
            if gen_example_method == 'normal':
                single_example = new_example
            elif gen_example_method == 'random_label':
                single_example = input_str + ' ' + random.choice(ans_str)
            elif gen_example_method == 'no_template':
                single_example = input_str.split(":")[1].split("\n")[0] + ' ' + ans
            elif gen_example_method == 'random_order':
                # random change the order of each word in the input string
                single_example = new_example
                words = single_example.split()
                random.shuffle(words)
                single_example = ' '.join(words)
            else:
                raise ValueError("Unknown demonstration example generation method!")
            single_example = single_example + example_separator
            demonstration_expample_list.append(single_example)
            
        if add_extra_query:  # add a random query at the end of demonstration
            extra_qeury, _, _ = self.apply_template(self.all_data[random.randint(0, len(self.all_data))])
            demonstration += extra_qeury
            demonstration_expample_list.append(extra_qeury)

        # check length of demonstration token
        encoded_inputs = tokenizer(demonstration, return_tensors="pt", padding=True, truncation=False)
        assert len(encoded_inputs['input_ids'][0]) < max_demonstration_tok_len, "Demonstration token length should be smaller than the maximum demonstration token length!"
        print(f"Generated {shot_num}-shot demonstration.")
        
        if return_data_index:
            return demonstration, demonstration_expample_list, sample_indexes
        else:
            return demonstration, demonstration_expample_list


    def get_dmonstration_template(self):
        """
        This function is used to provide template for demonstration, need to be implemented for each task.
        """
        raise NotImplementedError("Please provide the template for demonstration!")
    
    def get_task_instruction(self):
        """
        This function is used to provide task instruction, need to be implemented for each task.
        """
        raise NotImplementedError("Please provide the task instruction!")
    
    def apply_template(self, data):
        """
        This function is used to apply template to a given data, need to be implemented for each task.
        """
        raise NotImplementedError("Please provide how to apply template!")
    
    def print_data(self, indices):
        """
        This function is used to print data given indices.
        """
        if isinstance(indices, int):
            indices = [indices]
        for index in indices:
            print(self.all_data[index])

    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        return self.all_data[index]