from trainer.recbole_inference import inference, test_inference
import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='EASE-Jun-10-2023_14-11-23.pth')
    
    args = parser.parse_args()
    
    # inference(args.path)
    test_inference(args.path)