import os 
import pandas as pd 
import numpy as np
from itertools import combinations
import argparse


def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

def count_similiarity(df1, df2):
    df = pd.concat([df1,df2])
    df = df.groupby(['user','item']).size().reset_index(name='counts')
    df = df[df['counts'] > 1]
    return len(df) / len(df1) * 100

def main(args):
    files = os.listdir(args.file_path)
    
    for df1, df2 in combinations(files, 2):
        sim = count_similiarity(pd.read_csv(os.path.join(args.file_path, df1)), pd.read_csv(os.path.join(args.file_path, df2)))
        df1_name = df1.split('.')[0]
        df2_name = df2.split('.')[0]
        print(f'{df1_name} and {df2_name}: {sim:.2f}% similarity')
        
if __name__== '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='submission/')
    args = parser.parse_args()
    
    main(args)