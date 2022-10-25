import pandas as pd
import numpy as np
import glob
import codecs
import seaborn as s
from matplotlib import pyplot as plt

OCS_CUT_OFF = 12  # 強迫のCBCLカットライン。（-8点）以下は強迫なしとする。

# 最大表示列数の指定（ここでは50列を指定）
# pd.set_option('display.max_columns', 20)

# 第１期基礎データ
with codecs.open("/Volumes/Pegasus32R8/TTC/2022base_OC_PLE/180511AB基本セット（CBCL・SDQ・SMFQ）_200720.csv", "r", "Shift-JIS",
                 "ignore") as file1:
    df1 = pd.read_table(file1, delimiter=",", low_memory=False)
    print(df1.head(20))
    df1 = df1[["SAMPLENUMBER", "TTC_sex", "AA1age", "AE1BMI",
               "AEIQ", "AAFedu", "AAMedu", "AA79Fsep", "AB161MIQ"]]
    print(df1.head(20))

    base_1st = df1.replace(r'^\s+$', np.nan, regex=True)  # 空白をNaNに置換
    print("第１期基礎データ\n", base_1st)
    base_1st = base_1st.set_index("SAMPLENUMBER")


# プログラム2｜所定フォルダ内の「data*.xlsx」を取得
files = glob.glob('/Volumes/Pegasus32R8/TTC/2022base_OC_PLE/*.xlsx')

# プログラム3｜変数listを空リストで設定
ls = []

# プログラム4｜プログラム2で取得したエクセルを一つずつpandasデータとして取得
for file in files:
    d = pd.read_excel(file)
    # print(d)
    d = d.set_index("SAMPLENUMBER")
    # print(d)
    ls.append(d)

# プログラム5｜listに格納されたエクセルファイルをpandasとして結合
oc_ple = pd.concat(ls, axis=1, join='inner')
print(oc_ple)


# 第２期のOC
# 強迫の人数(cut off 5以上)
oc_2nd = oc_ple[["BB39", "BB56", "BB57", "BB73", "BB83", "BB95", "BB96", "BB116"]]
print("第２期のOC\n", oc_2nd)
print("NaN個数\n", oc_2nd.isnull().sum())
oc_2nd = oc_2nd.dropna(how='any')  # ★強迫症状の項目にNaNを含むもの削除
oc_2nd["OCS_sum"] = oc_2nd.sum(axis=1)
print("OCS合計点\n", oc_2nd)

oc_pos = (oc_2nd["OCS_sum"] > OCS_CUT_OFF)
print("第２期にOCあり\n", oc_pos)
print("強迫症状カットオフ以上\n", oc_pos.sum())  # 5点以上だと115人

# OCS13以上を1、12以下を0にする
oc_2nd["OCS_0or1"] = (oc_2nd["OCS_sum"] > OCS_CUT_OFF) * 1
print("OCS_0or1\n", oc_2nd)  # 2733行

# AQ
aq_2nd = oc_ple.filter(regex='^(BB12|BB13)', axis=1)
print(aq_2nd)


# 第３期PLEのデータフレーム
ple = oc_ple.filter(like='_1', axis=1)
print("抽出：\n", ple)  # PLEのうち、頻度を聞く項目のみ

ple_3rd = ple.filter(like='CD', axis=1)
ple_3rd = ple_3rd.drop('CD70_1', axis=1)  # PLEのうち、第３期のみ
print("第３期PLE：\n", ple_3rd)


# 第１期にPLEがなかった群を抽出
df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv/columns_NAN_under_500.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df.head())
# print("第１期データのNaN個数\n", df.isnull().sum())

# 1回目調査でPLEなしを抽出したい
df_PLE = df[["AD57", "AD58", "AD59", "AD60", "AD61", "AD62", "AD63"]]
print("df_PLE\n", df_PLE)
ple_1st_pos = (df_PLE == 1.0)
print("ple_1st_pos\n", ple_1st_pos)  # "Yes, definitely"
print("1回目調査でPLEが「Yes, definitely」\n", ple_1st_pos.sum())

ple_1st_neg = ((df_PLE == 3.0) | (df_PLE == 2.0))
print("ple_1st_neg\n", ple_1st_neg)
print("1回目調査でPLEなしの人数\n", ple_1st_neg.sum())
print("1回目調査でPLEなし\n", df_PLE[ple_1st_neg])

ple_neg_oc = pd.merge(df_PLE[ple_1st_neg], oc_2nd, left_index=True, right_index=True)  # 第１期にpleなしのうち、OCにNaNなし
print(ple_neg_oc.head())

cols_to_use = df.columns.difference(ple_neg_oc.columns)  # baseにある項目を検出
print(cols_to_use)
df4grf = ple_neg_oc.join([df[cols_to_use], aq_2nd, ple_3rd], how='inner')
print(df4grf)
df4grf.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_alldata/data4grf.csv")  # 共変数から文字列を含む列、PLEの_2、baseに含まれる列を削除したもの

print("NaN個数\n", df4grf["OCS_0or1"].isnull().sum())

