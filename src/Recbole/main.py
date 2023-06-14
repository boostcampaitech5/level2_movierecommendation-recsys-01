from trainer.recbole_train import *
from trainer.recbole_inference import inference
from utils.preprocess_data import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ease')
    parser.add_argument('--model_type', type=str, default='general')
    parser.add_argument('--make_yaml', type=bool, default=True)
    
    args = parser.parse_args()
    
    if args.model_type=='general':
        if args.make_yaml:
            make_general_yaml(args.model_name)
        general_train(args.model_name)
        
    elif args.model_type=='context':
        if args.make_yaml:
            make_context_yaml(args.model_name)
        context_train(args.model_name)
    