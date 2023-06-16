import pandas as pd 
import os

def make_general_dataset(path='../../data/train/'):
    
    df = pd.read_csv(os.path.join(path,'train_ratings.csv'))
    
    df.columns=['user_id:token','item_id:token','timestamp:float']
    outpath = f"dataset/general_train"
    
    os.makedirs(outpath, exist_ok=True)
    df.to_csv(os.path.join(outpath,"general_train.inter"),sep='\t',index=False)
    
def make_context_dataset(path='../../data/train/'):
    
    train = pd.read_csv(os.path.join(path, 'train_ratings.csv'))
    directors = pd.read_csv(os.path.join(path, "directors.tsv"), sep="\t")
    genres = pd.read_csv(os.path.join(path, "genres.tsv"), sep="\t")
    titles = pd.read_csv(os.path.join(path, "titles.tsv"), sep="\t")
    writers = pd.read_csv(os.path.join(path, "writers.tsv"), sep="\t")
    years = pd.read_csv(os.path.join(path, "years.tsv"), sep="\t")
    
    def make_feature_sequence(x):
        x = list(set(x))
        y = ""
        for item in x:
            y += str(item + " ")
        return y.rstrip()
    
    writers_seq = writers.groupby("item")['writer'].apply(make_feature_sequence)
    genres_seq = genres.groupby("item")['genre'].apply(make_feature_sequence)
    director_seq = directors.groupby('item')['director'].apply(make_feature_sequence)

    train_df = pd.merge(train, writers_seq, on="item",how='left')
    train_df = pd.merge(train_df, genres_seq, on="item",how='left')
    train_df = pd.merge(train_df, director_seq, on="item",how='left')
    train_df = pd.merge(train_df, years, on="item",how='left')
    
    train_df = train_df.sort_values('user')
    
    train_data = train_df[['user', 'item', 'time']].reset_index(drop=True)
    user_data = train_df[['user']].reset_index(drop=True)
    item_data = train_df[['item', 'year', 'writer', 'genre', 'director']].drop_duplicates(subset=['item']).reset_index(drop=True)
    
    train_data.columns=['user_id:token', 'item_id:token', 'timestamp:float']
    user_data.columns=['user_id:token']
    item_data.columns=['item_id:token', 'year:token', 'writer:token_seq', 'genre:token_seq', 'director:token_seq']
    
    outpath = f"dataset/context_train"

    os.makedirs(outpath, exist_ok=True)

    print("Dump Start")

    # 데이터 파일 저장
    train_data.to_csv(os.path.join(outpath,"context_train.inter"),sep='\t',index=False)
    user_data.to_csv(os.path.join(outpath,"context_train.user"),sep='\t',index=False)
    item_data.to_csv(os.path.join(outpath,"context_train.item"),sep='\t',index=False)
    
    print("Dump Complete")

    
def make_general_yaml(model_name):
    
    yaml_data = """
    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    TIME_FIELD: timestamp

    load_col:
        inter: [user_id, item_id, timestamp]
    """
    outpath = f"general_yaml"
    
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(outpath,f"{model_name}.yaml"),"w") as f:
        f.write(yaml_data)
        

def make_context_yaml(model_name):
    yaml_data = """
    field_separator: '\t'
    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    TIME_FIELD: timestamp

    load_col:
        inter: [user_id, item_id, timestamp]
        user: [user_id]
        item: [item_id, year, genre, writer, director]

    train_neg_sample_args:
        uniform: 10
        
    eval_args:
        split: {'RS': [8, 1, 1]}
        group_by: user
        order: RO
        mode: full
    metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision', 'MAP']
    topk: 10
    valid_metric: Recall@10
    """
    
    outpath = f"context_yaml"
    
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(outpath,f"{model_name}.yaml"),"w") as f:
        f.write(yaml_data)
        
def make_sequence_yaml(model_name):
    
    yaml_data="""
    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    TIME_FIELD: timestamp

    load_col:
        inter: [user_id, item_id, timestamp]
        
    train_neg_sample_args: ~
    ITEM_LIST_LENGTH_FIELD: item_length
    LIST_SUFFIX: _list
    MAX_ITEM_LIST_LENGTH: 50
    """
    outpath = f"sequence_yaml"
    
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(outpath,f"{model_name}.yaml"),"w") as f:
        f.write(yaml_data)
        