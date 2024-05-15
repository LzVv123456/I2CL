from .basetask import BaseTask

from .sst2 import SST2
from .dbpedia import DBPedia
from .sst5 import SST5
from .trec import TREC
from .agnews import AGNews
from .subj import Subj
from .rotten_tomatoes import RottenTomatoes
from .hate_speech18 import HateSpeech18
from .emo import EMO


target_datasets = {
    'agnews': AGNews,
    'dbpedia': DBPedia,
    'sst5': SST5,
    'trec': TREC,
    'sst2': SST2,
    'subj': Subj,
    'mr': RottenTomatoes,
    'hate_speech18': HateSpeech18,
    'emo': EMO,
}

dataset_dict = {}
dataset_dict.update(target_datasets)

def get_dataset(dataset, *args, **kwargs) -> BaseTask:
    return dataset_dict[dataset](task_name=dataset, *args, **kwargs)