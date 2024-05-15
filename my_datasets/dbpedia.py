try:
    from . import BaseTask
except:
    from basetask import BaseTask
# from basetask import BaseTask
from datasets import load_dataset


class DBPedia(BaseTask):
    def __init__(self, split='train', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "classification"
        # set split name
        if split in ['train', 'validation']:
            load_split = 'train'
        else:
            load_split = 'test'

        print(f"Loading {load_split} data from DBPedia ...")
        # class_num
        self.class_num = 14
        # load dataset
        self.dataset = load_dataset('fancyzhx/dbpedia_14', split=load_split, keep_in_memory=True)
        # get all data
        self.all_data = [data for data in self.dataset]
        # get all labels
        self.all_labels = self.get_all_labels()
        # random sample data
        if self.max_data_num is not None:
            self.random_sample_data(self.max_data_num)
        # print a few examples
        print(f'Dataset lengh is {len(self.all_data)}')
        print("Example data:")
        self.print_data([0])

    def get_dmonstration_template(self):
        template = {
            'input': 'Input: {text}\nLabel:',
            'ans': '{answer}',
            'options': ['company', 'school', 'artist', 'athlete', 
                        'politics', 'transportation', 'building', 
                        'nature', 'village', 'animal', 'plant',
                        'album', 'film', 'book'],
            'format': ['Input:', 'Label:']
        }
        return template

    def get_task_instruction(self):
        task_instruction = "Classify the sentiment of the sentence into one of the categories: positive or negative.\n\n"
        return task_instruction
      
    def apply_template(self, data):
        """
        PS: label should always be an integer and can be used to index the options
        """
        template = self.get_dmonstration_template()
        input_template = template['input']
        ans_template = template['ans']
        options = template['options']
        input_str = input_template.replace("{text}", data["content"])
        # answers can have multiple options and is a list
        answer_str = [ans_template.replace("{answer}", options[i]) for i in range(len(options))]
        label = data["label"]
        return input_str, answer_str, label
    
    def get_all_labels(self):
        return [data["label"] for data in self.all_data]