import sys
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import os
import argparse
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    
    return v


def main(args):
    files = os.listdir(args.file_path)
    files.sort()
    
    df_list = [pd.read_csv(os.path.join(args.file_path, i)) for i in files]
    user_list = df_list[0]['user'].unique()
    df_len = len(df_list)
    ensemble_ratio = args.weight
    
    result = []
    tbar = tqdm(user_list,desc='Ensemble')
    
    for user in tbar:
        temp = defaultdict(float)
        for idx in range(df_len):
            items = df_list[idx][df_list[idx]['user']==user]['item'].values
            
            for item_idx,item in enumerate(items):
                temp[item] += ensemble_ratio[idx] * (1 - item_idx/len(items))
                
        for key, _ in sorted(temp.items(),key=lambda x:x[1],reverse=True)[:10]:
            result.append((user,key))
    
    submission = pd.DataFrame(result, columns=['user', 'item'])
    submission.to_csv('submission.csv', index=False)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--file_path', type=str, default='for_ensemble/')
    parser.add_argument('--weight',type=arg_as_list, default=[])
    
    args = parser.parse_args()
    main(args)