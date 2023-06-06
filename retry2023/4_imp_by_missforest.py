# from multiprocessing import cpu_count
from multiprocessing import cpu_count

import pandas as pd
from missingpy import MissForest
import sys
import sklearn.neighbors._base
# sklearnのバージョンによって、.baseが_.base となったことによるエラーに対処
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base


df0 = pd.read_table("/Volumes/Pegasus32R8/TTC/2023retry/columns_NAN_under_5percent.csv",
                    delimiter=",")
df0 = df0.set_index("SAMPLENUMBER")
print(df0)

# Make an instance and perform the imputation
imputer = MissForest(criterion='squared_error', max_features=1.0, n_jobs=int(cpu_count() * 4 / 5))

# https://pypi.org/project/missingpy/
df_imputed = imputer.fit_transform(df0)
print("df_imputed\n", df_imputed)

# 各列を整数に丸める（身長、体重も丸め）
df_imputed = df_imputed.round().astype(int)

# 第3, 第4期のPLE（欠損値あり）と統合
df0[df0.columns.values] = df_imputed
df0.to_csv("/Volumes/Pegasus32R8/TTC/2023retry/df4grf_all_imputed.csv")

print("set_indexされているか確認\n", df0)
