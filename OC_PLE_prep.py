import pandas as pd
import numpy as np
import glob
import codecs
import seaborn as s
from matplotlib import pyplot as plt

OCS_CUT_OFF = 12  # 強迫のCBCLカットライン。（-8点）以下は強迫なしとする。

# 最大表示列数の指定（ここでは50列を指定）
# pd.set_option('display.max_columns', 20)

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
oc_2nd = oc_2nd.dropna(how='any')  # NaNを削除
oc_2nd["OCS_sum"] = oc_2nd.sum(axis=1)  # , skipna=False)
print("OCS合計点\n", oc_2nd)

oc_pos = (oc_2nd["OCS_sum"] > OCS_CUT_OFF)
print("第２期にOCあり\n", oc_pos)
print("強迫症状カットオフ以上\n", oc_pos.sum())  # 5点以上だと115人

# OCS13以上を1、12以下を0にする
oc_2nd["OCS_0or1"] = (oc_2nd["OCS_sum"] > OCS_CUT_OFF) * 1
print(oc_2nd)  # 2733行


# 第３期PLEのデータフレーム
ple = oc_ple.filter(like='_1', axis=1)
print("抽出：\n", ple)  # PLEのうち、頻度を聞く項目のみ

ple_3rd = ple.filter(like='CD', axis=1)
ple_3rd = ple_3rd.drop('CD70_1', axis=1)  # PLEのうち、第３期のみ
print("第３期PLE：\n", ple_3rd)


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


# 第１期にPLEがなかった群を抽出
"""
df = pd.read_table(""/Volumes/Pegasus32R8/TTC/2022csv/columns_NAN_under_500.csv"", delimiter=",")  # このファイルをiMacでも使えるように★
print(df.describe())
print(df.head(30))
df = df.set_index("SAMPLENUMBER")
print(df.head(30))

# 列ごとにNaNをいくつ含むか
print("NaN個数\n", df.isnull().sum())

# 1回目調査でPLEありの人数
df_PLE = df[["AD57", "AD58", "AD59", "AD60", "AD61", "AD62", "AD63"]]
print("df_PLE\n", df_PLE)
# df_PLE_pos = ((df_PLE == 1.0) | (df_PLE == 2.0))
df_PLE_pos = (df_PLE == 1.0)
print("df_PLE_pos\n", df_PLE_pos)  # "Yes, definitely"
print("1回目調査でPLEが「Yes, definitely」\n", df_PLE_pos.sum())

df_PLE_neg = ((df_PLE == 3.0) | (df_PLE == 2.0))
print("df_PLE_neg\n", df_PLE_neg)
print("1回目調査でPLEなし\n", df_PLE_neg.sum())

# 強迫の人数(cut off 5以上)
df_OCS = df[["BB39", "BB56", "BB57", "BB73", "BB83", "BB95", "BB96", "BB116"]]
print("df_OCS\n", df_OCS)
print("NaN個数\n", df_OCS.isnull().sum())
df_OCS = df_OCS.dropna(how='any')  # NaNを削除
df_OCS["OCS_sum"] = df_OCS.sum(axis=1)  # , skipna=False)
print("OCS合計点\n", df_OCS)

df_OCS_pos = (df_OCS["OCS_sum"] > OCS_CUT_OFF)  # 5点以上だと115人
print("df_OCS_pos\n", df_OCS_pos)
print("強迫症状カットオフ以上\n", df_OCS_pos.sum())

print("df_OCSカットオフ以上以上\n",
      pd.merge(df_PLE, df_OCS[df_OCS_pos], left_index=True, right_index=True))  # df_OCSで13点以上を抽出
df0 = pd.merge(df_PLE[df_PLE_neg], df_OCS, left_index=True, right_index=True)  # かつ、PLEなし
print("PLEなし\n", df0.dropna(how='any'))  # NaNを削除
df0 = df0.dropna(how='any')
df1 = pd.merge(df_PLE[df_PLE_neg], df_OCS[df_OCS_pos], left_index=True, right_index=True)  # df_OCS13以上でかつPLEなし
print("df_OCSカットオフ以上でかつPLEなし\n", df1.dropna(how='any'))  # NaNを削除

# OCS13以上を1、12以下を0にする
df_OCS["OCS_0or1"] = (df_OCS["OCS_sum"] > OCS_CUT_OFF) * 1
print(df_OCS)  # 2733行

cols_to_use = df.columns.difference(df0.columns)  # baseにある項目を検出
df_grf = df0.join([df[cols_to_use], df_OCS["OCS_0or1"]], how='inner')
print(df_grf)
# df_grf.to_csv("TTC2022_ple_naive.csv")  # 共変数から文字列を含む列、PLEの_2、baseに含まれる列を削除したもの

print("NaN個数\n", df_grf.isnull().sum())
# s=df_grf.isnull().sum()
# print(s.head(264))
# s.to_csv("TTC2022_grf_min_NaN.csv")
# ★他の項目が手に入ったら、ここでdf_grfにマージする！！！★
"""