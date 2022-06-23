import os
import numpy as np
import pandas as pd
import glob
import multiprocessing as mp

from tqdm import tqdm

def bytes_size(path):

    if isinstance(path, str):
        path = [path]

    return np.sum([os.path.getsize(file) for file in path])

def process_file(file, update):
    type, symbol, date = file.split('/')[-3:]

    df: pd.DataFrame = pd.read_csv(file, header = 0, names = ['id', 'side', 'amount', 'price', 'timestamp'])
    df = df.drop_duplicates(subset = ['id', 'price', 'amount', 'timestamp'], keep = 'first')
    df.to_csv(file, compression={ 'method': 'zip', 'archive_name': f'{symbol}-{type}-{date}.csv' })

    update(bytes_size(file))

def preprocess_files():

    path = os.path.join('data', 'okx')

    files = glob.glob(os.path.join(path, 'trades', '*/*'))
    files.sort()

    sz = bytes_size(files)

    with tqdm(total=sz) as pbar:

        pool = mp.Pool(mp.cpu_count())
        jobs = []

        for file in files:
            jobs.append(pool.apply_async(process_file, (file, pbar.update)))

        for job in jobs:
            job.get()

preprocess_files()