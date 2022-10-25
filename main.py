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
