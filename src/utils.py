import multiprocessing
from multiprocessing import Pool

import pandas as pd
import numpy as np

def paralize_df(df, method, num_workers=None):
    # parallelize execution
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    unuque_instalation_ids = df.installation_id.unique()
    instalation_ids_list = np.array_split(unuque_instalation_ids, num_workers)
    df_split = [df.loc[df.installation_id.isin(instalation_ids)] for instalation_ids in instalation_ids_list]
    pool = Pool(num_workers)
    df = pd.concat(pool.map(method, df_split))
    pool.close()
    pool.join()
    return df