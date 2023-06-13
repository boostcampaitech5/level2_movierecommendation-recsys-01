from trainer.recbole_train import train
from trainer.recbole_inference import inference
from utils.preprocess_data import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ease')
    
    args = parser.parse_args()
    
    make_yaml(args.model_name)
    train(args.model_name)
    
    