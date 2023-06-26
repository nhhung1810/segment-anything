from operator import is_
import os
from glob import glob
import pandas as pd
import shutil

from scripts.utils import make_directory

FILENAMES = list(glob("runs/*.csv"))
CONFIG_PATH = "config-230619-213933.csv"
CTX = {}
PATH_DICT = {}

def cross_check(dsts, logdir, actual_paths):
    result = set()
    for dst in dsts:
        for lg, ap in zip(logdir, actual_paths):
            if lg in dst: 
                result.add(dst)
                CTX[dst] = lg
                PATH_DICT[dst] = ap
        pass
    return list(result)

def separate(name):
    checknum = int(
        name.split(CTX[name])[-1].replace("-", "").replace("_", "")
        )
    exp_name = str(
        name.split(CTX[name])[0].replace("-", "").replace("_", "")
        )
    return checknum, exp_name
    
def keep_max(x: pd.core.frame.DataFrame):
    best_mean = x.iloc[[x['DSC_mean'].argmax()]]
    best_liver = x.iloc[[x['DSC_1'].argmax()]]
    best_gall = x.iloc[[x['DSC_9'].argmax()]]
    return pd.concat([best_mean, best_liver, best_gall]).reset_index(drop='index')

if __name__ == "__main__":
    config_df = pd.read_csv(CONFIG_PATH)
    logdirs = [os.path.basename(f) for f in config_df['logdir'].to_list()]
    actual_paths = [f for f in config_df['path'].to_list()]

    join_df = []
    for filename in FILENAMES:
        df = pd.read_csv(filename)
        names = df['Name'].tolist()
        u = cross_check(dsts=names, logdir=logdirs, actual_paths=actual_paths)
        filter_df = df[df['Name'].apply(lambda x: x in u)]
        filter_df = filter_df[filter_df['DSC_2'] < 0.2]
        filter_df['fullname'] = filter_df['Name'].apply(lambda x: CTX[x])
        filter_df['path'] = filter_df['Name'].apply(lambda x: PATH_DICT[x])
        filter_df['exp'] = filter_df['Name'].apply(lambda x: separate(x)[1])
        filter_df['ckpt'] = filter_df['Name'].apply(lambda x: separate(x)[0])
        filter_df = filter_df.groupby(by='fullname').apply(keep_max)
        join_df.append(filter_df)
        pass    

    join_df = pd.concat(join_df)
    join_df = join_df[join_df['DSC_2'] < 0.2]
    new_folder = "runs/transfer/"
    for entries in join_df.iterrows():
        path = entries[-1]['path']
        ckpt = entries[-1]['ckpt']
        fullpath = os.path.join(path, f"model-{ckpt}.pt")
        if not os.path.exists(fullpath): continue
        new_path = os.path.join(new_folder, os.path.basename(os.path.dirname(path)), os.path.basename(fullpath))
        make_directory(new_path, is_file=True)
        shutil.copyfile(fullpath, new_path)
        # print(entries)
    # join_df = join_df[['Name', 'ckpt', 'DSC_mean', 'DSC_1', 'DSC_9']]
    # join_df.to_csv('test.csv')
    pass