import pandas as pd
import numpy as np
from missingpy import MissForest

df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_alldata/data4grf.csv", delimiter=",")
df = df.drop(["Unnamed: 0"], axis=1)
print(df.head())  # 968 columns

numeric_columns = [colname for colname in df.columns if df[colname].dtype == np.float]  # 数値のみ抽出
df = df[numeric_columns]
print(df.head())  # 945 columns

# Make an instance and perform the imputation
imputer = MissForest(criterion='squared_error', max_features=1.0)
df_imputed = imputer.fit_transform(df)
print("df_imputed\n", df_imputed)
df[df.columns.values] = df_imputed
df = df.round().astype(int)  # 各列を整数に丸める（身長、体重も丸め）
df.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_alldata/base_ple_imputed.csv")
print(df)
