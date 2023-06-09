import os 
import pandas as pd 
import numpy as np 
import torch 

from recbole.model.general_recommender.bpr import BPR
from recbole.model.general_recommender.ease import EASE

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, Interaction
from recbole.utils import init_logger, get_trainer, get_model, init_seed, set_color
from recbole.quick_start import load_data_and_model, run_recbole
from recbole.utils.case_study import full_sort_topk

from logging import getLogger

def train(model_name='ease'):
    logger = getLogger()
    
    config = Config(model=model_name.upper,dataset='recbole_train',config_file_list=[f"{model_name}.yaml"])
    config['show_progress'] = False 
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    logger.info(config)