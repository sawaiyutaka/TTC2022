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
files = glob.glob('/Volumes/Pegasus32R8/TTC/2022rawdata/a*.xls*')

# プログラム3｜変数listを空リストで設定
lst = []

# プログラム4｜プログラム2で取得したエクセルを一つずつpandasデータとして取得
for file in files:
    d = pd.read_excel(file)
    # print(d)
    d = d.set_index("SAMPLENUMBER")
    # print(d)
    lst.append(d)

# プログラム5｜listに格納されたエクセルファイルをpandasとして結合
df1 = pd.concat(lst, axis=1, join='inner')
print(df1)

# 解析シートのあるエクセルの読み込み
files = glob.glob('/Volumes/Pegasus32R8/TTC/2022rawdata/1*.xls*')

# プログラム3｜変数listを空リストで設定
lst2 = []

# プログラム4｜プログラム2で取得したエクセルを一つずつpandasデータとして取得
for file in files:
    d = pd.read_excel(file, sheet_name='解析')
    print(file)
    d = d.set_index("SAMPLENUMBER")
    d = d.filter(regex='Imp$', axis=1)  # Impで終わる列＝欠損値１以下なら平均値で補完
    # print(d)
    lst2.append(d)

# プログラム5｜listに格納されたエクセルファイルをpandasとして結合
df2 = pd.concat(lst2, axis=1, join='inner')
print(df2)

df = pd.concat([df1, df2], axis=1, join='inner')
print(df)

# ペット飼育のデータフレーム
d = pd.read_excel("/Volumes/Pegasus32R8/TTC/2022rawdata_copy/171114A子ペット.xlsx", sheet_name='作業')
print(d)
d = d.set_index("SAMPLENUMBER")
d0 = d[["AE10"]]
print(d0)
d1 = d.filter(regex='Kind$', axis=1)  # Impで終わる列＝欠損値１以下なら平均値で補完
print(d1)
d2 = 1 - d1.isna() * 1
print(d2)
# d2.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/pet_kind.csv")

d3 = pd.merge(d0, d2, left_index=True, right_index=True)
print(d3)
print(d3["AE10"].isnull().sum())

# 第1期のデータフレーム結合
df = pd.merge(df, d3, left_index=True, right_index=True)
df = df.loc[:, ~df.columns.duplicated()]
print("重複を削除\n", df)

# プログラム6｜エクセルファイルを書き出す
df.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_boruta/TTC2022_1st_all.csv")

