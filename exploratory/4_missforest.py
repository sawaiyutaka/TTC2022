from multiprocessing import cpu_count

import pandas as pd
import numpy as np
from missingpy import MissForest
import sys
import sklearn.neighbors._base
# sklearnのバージョンによって、.baseが_.base となったことによるエラーに対処
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base


df0 = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_boruta/columns_NAN_under_92.csv",
                    delimiter=",")
# df = df.drop(["Unnamed: 0"], axis=1)
df0 = df0.set_index("SAMPLENUMBER")
print(df0.head())

df = df0.drop(columns=["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"])
df = df.drop(columns=["DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1", "DD69_1", "DD70_1", "DD71_1", "DD72_1"])

# Make an instance and perform the imputation
imputer = MissForest(criterion='squared_error', max_features=1.0, n_jobs=int(cpu_count() / 2))
"""
                     max_iter=10, decreasing=False, missing_values=np.nan,
                     copy=True, n_estimators=2000, 
                     max_depth=None, min_samples_split=2, min_samples_leaf=1,
                     min_weight_fraction_leaf=0.0, max_features=200,
                     max_leaf_nodes=None, min_impurity_decrease=0.0,
                     bootstrap=True, oob_score=False, random_state=None,
                     verbose=0, warm_start=False, class_weight=None)
"""
# https://pypi.org/project/missingpy/


df_imputed = imputer.fit_transform(df)
print("df_imputed\n", df_imputed)
df0[df.columns.values] = df_imputed
df0 = df0.round().astype(int)  # 各列を整数に丸める（身長、体重も丸め）
df0.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_boruta/imputed.csv")
print("set_indexは？\n", df)
