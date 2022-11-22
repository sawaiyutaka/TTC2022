import pandas as pd
import numpy as np
from missingpy import MissForest

df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_outcome/data4grf.csv", delimiter=",")
# df = df.drop(["Unnamed: 0"], axis=1)
df = df.set_index("SAMPLENUMBER")
print(df.head())

numeric_columns = [colname for colname in df.columns if df[colname].dtype == float]  # 数値のみ抽出
df = df[numeric_columns]
print(df.head())

# Make an instance and perform the imputation
imputer = MissForest(max_iter=10, decreasing=False, missing_values=np.nan,
                     copy=True, n_estimators=2000, criterion='squared_error',
                     max_depth=None, min_samples_split=2, min_samples_leaf=1,
                     min_weight_fraction_leaf=0.0, max_features=200,
                     max_leaf_nodes=None, min_impurity_decrease=0.0,
                     bootstrap=True, oob_score=False, n_jobs=10, random_state=None,
                     verbose=0, warm_start=False, class_weight=None)
# https://pypi.org/project/missingpy/
# n_estimators=2000, max_features=200は別のプログラムのgrid searchから当てはめた
# n_jobsは手控えて10とする

df_imputed = imputer.fit_transform(df)
print("df_imputed\n", df_imputed)
df[df.columns.values] = df_imputed
df = df.round().astype(int)  # 各列を整数に丸める（身長、体重も丸め）
df.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/base_ple_imputed.csv")
print("set_indexは？\n", df)
