import pandas as pd
import numpy as np
import openpyxl
import glob
import os
import codecs

# プログラム2｜所定フォルダ内の「data*.xlsx」を取得
files = glob.glob('*.xls*')

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

# プログラム6｜エクセルファイルを書き出す
df1.to_csv("test1.csv")
