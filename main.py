import pandas as pd
import numpy as np
import openpyxl
import glob
import os
import codecs

# 現在の最大表示列数の出力
# pd.get_option("display.max_columns")

# 最大表示列数の指定（ここでは50列を指定）
# pd.set_option('display.max_columns', 50)


# 第１期量的データ全部

# プログラム2｜所定フォルダ内の「data*.xlsx」を取得
files=glob.glob('/Volumes/Pegasus32R8/TTC/2022rawdata/*.xlsx')

# プログラム3｜変数listを空リストで設定
list = []

# プログラム4｜プログラム2で取得したエクセルを一つずつpandasデータとして取得
for file in files:
    d = pd.read_excel(file)
    print(d)
    d = d.set_index("SAMPLENUMBER")
    print(d)
    list.append(d)

# プログラム5｜listに格納されたエクセルファイルをpandasとして結合
df = pd.concat(list, axis=1, join='inner')

# プログラム6｜エクセルファイルを書き出す
df.to_csv("/Volumes/Pegasus32R8/TTC/2022csv/TTC2022_1st_all.csv")
# df.to_csv("/Volumes/Pegasus32R8/TTC/2022csv/TTC2022_1st_all.csv")

"""
# 上のCSVから、文字列を含む列、PLEの_2、baseに含まれる列を削除したもの
df0 = pd.read_table("TTC2022_variates_minimum.csv", delimiter=",")
print(df0.head(50))
# df0 = df0.set_index("SAMPLENUMBER")

import codecs

with codecs.open("ab_base.csv", "r", "Shift-JIS", "ignore") as file1:
    df1 = pd.read_table(file1, delimiter=",")  # , names=col_names)
    print(df1.head(20))
    df1 = df1[["SAMPLENUMBER", "AA1age", "AC1", "AE1", "AE2", "AE1BMI",
               "AEIQ", "AAFedu", "AAMedu", "AB195", "AA79Fsep",
               "AB161MIQ"]]
    print(df1.head(20))
    # df1 = df1.set_index("SAMPLENUMBER")

# cols_to_use = df1.columns.difference(df.columns)
df2 = pd.merge(df0, df1, on="SAMPLENUMBER", how='outer')
df2 = df2.replace(r'^\s+$', np.nan, regex=True)
# df2 = df0.join(df1)
print(df2)

print("df1NANcheck\n", df1.isnull().sum())
# df2.to_csv("TTC2022_base_minimum.csv")
"""