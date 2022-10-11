import pandas as pd
import numpy as np
import openpyxl

# 現在の最大表示列数の出力
# pd.get_option("display.max_columns")

# 最大表示列数の指定（ここでは50列を指定）
# pd.set_option('display.max_columns', 50)

"""
# 申請したデータ全部（AB基本セットは含めず）
df_aa = pd.read_excel('aa_n4477.xlsx', index_col=0)
print(df_aa.head(30))

df_ab = pd.read_excel('ab_n4477.xlsx', index_col=0)
print(df_ab.head(30))

df_ad = pd.read_excel('ad_n4470.xls', index_col=0)
print(df_ad.head(30))

df_ae = pd.read_excel('ae_n4478.xls', index_col=0)
print(df_ae.head(30))

df_bb = pd.read_excel('bb_n3007.xlsx', index_col=0)
print(df_bb.head(30))

df_bd = pd.read_excel('bd_n3007.xls', index_col=0)
print(df_bd.head(30))

df_cd = pd.read_excel('cd_n3171.xlsx', index_col=0)
print(df_cd.head(30))

df_db = pd.read_excel('db_n3171.xlsx', index_col=0)
print(df_db.head(30))

df_dd = pd.read_excel('dd_n3171.xlsx', index_col=0)
print(df_dd.head(30))

df = df_aa.join([df_ab, df_ad, df_ae, df_bb, df_bd, df_cd, df_db, df_dd])
print(df.head(30))
"""
# df.to_csv("TTC2022_all.csv")


# df = pd.read_table("TTC2022_full_number_only.csv", delimiter=",")
# print(df.head())

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
