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
with codecs.open("180511AB基本セット（CBCL・SDQ・SMFQ）_200720.csv", "r", "Shift-JIS",
                 "ignore") as file1:
    df1 = pd.read_table(file1, delimiter=",", low_memory=False)
    df1 = df1.set_index("SAMPLENUMBER")
    print(df1.head())
    df1 = df1[["TTC_sex", "AE1BMI",
               "AEIQ", "AAFedu", "AAMedu", "AA79Fsep", "AB161MIQ", "AA127Respondent"]]
    print(df1.head())

    base_1st = df1.replace(r'^\s+$', np.nan, regex=True)  # 空白をNaNに置換
    print("第１期基礎データ\n", base_1st)

oc_ple = pd.read_table("test1.csv", delimiter=",")
oc_ple = oc_ple.set_index("SAMPLENUMBER")
print(oc_ple)

# 第1期のデータ
first = oc_ple.filter(like='A', axis=1)

# 第２期のOC
# 強迫の人数(cut off 5以上)
oc_2nd = oc_ple[["BB39", "BB56", "BB57", "BB73", "BB83", "BB95", "BB96", "BB116"]]
print("第２期のOC\n", oc_2nd)
print("NaN個数\n", oc_2nd.isnull().sum())
oc_2nd = oc_2nd.dropna(how='any')  # ★強迫症状の項目にNaNを含むもの削除
oc_2nd["OCS_sum"] = oc_2nd.sum(axis=1)
print("第２期にOC欠損値なし\n", oc_2nd)

oc_pos = (oc_2nd["OCS_sum"] > OCS_CUT_OFF)
print("強迫症状カットオフ以上\n", oc_pos.sum())  # 5点以上だと115人

# OCS13以上を1、12以下を0にする
oc_2nd["OCS_0or1"] = (oc_2nd["OCS_sum"] > OCS_CUT_OFF) * 1.0
print("OCS_0or1\n", oc_2nd)  # 第２期にOC欠損値なしは2733行

# ★第1期にOCがない人を抽出★
oc_1st = oc_ple[["AB71", "AB87", "AB88", "AB104", "AB114", "AB126", "AB127", "AB145"]]
print("第1期のOC\n", oc_1st)
print("NaN個数\n", oc_1st.isnull().sum())
oc_1st = oc_1st.dropna(how='any')  # ★強迫症状の項目にNaNを含むもの削除
oc_1st["OCS_1st_sum"] = oc_1st.sum(axis=1)
print("第1期にOC欠損値なし\n", oc_1st)

oc_naive = (oc_1st["OCS_1st_sum"] <= OCS_CUT_OFF)
print("第1期強迫症状カットオフ未満\n", oc_naive.sum())


# AQ
aq_2nd = oc_ple[["BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132"]]
print("第2期AQ素点", aq_2nd)

# 第3期PLEのデータフレーム
ple = oc_ple.filter(like='_1', axis=1)
print("抽出：\n", ple)  # PLEのうち、頻度を聞く項目のみに絞りたい

ple_3rd = ple.filter(like='CD', axis=1)  # PLEのうち、第3期のみ
ple_3rd = ple_3rd.drop('CD70_1', axis=1)  # PLE第3期のうち、頻度以外の項目を削除
print("第3期PLE：\n", ple_3rd)

# 第4期PLEのデータフレーム
ple = oc_ple.filter(like='_1', axis=1)
print("抽出：\n", ple)  # PLEのうち、頻度を聞く項目のみに絞りたい

ple_4th = ple.filter(like='DD', axis=1)  # PLEのうち、第4期のみ
ple_4th = ple_4th.drop('DD77_1', axis=1)  # PLE第4期のうち、頻度以外の項目を削除
print("第4期PLE：\n", ple_4th)

# 第１期にPLEがなかった群を抽出

# 1回目調査でPLEなしを抽出したい
df_PLE = oc_ple[["AD57", "AD58", "AD59", "AD60", "AD61", "AD62", "AD63"]]
print("df_PLE\n", df_PLE)
print("NaN個数\n", df_PLE.isnull().sum(axis=0))
ple_1st_pos = (df_PLE != 1.0)
print("'Yes, definitely'でないところをTRUE\n", ple_1st_pos)
# print("1回目調査でPLEが「Yes, definitely」\n", ple_1st_pos.sum())
# ple_negと排反になることを確認

ple_1st_neg = ((df_PLE == 3.0) | (df_PLE == 2.0))
print("ple_1st_neg\n", ple_1st_neg)
print("1回目調査でPLEなしの人数\n", ple_1st_neg.sum())
print(df_PLE[ple_1st_neg])
ple_neg = df_PLE[ple_1st_neg].dropna(how='any')  # ★第１期に「あったかもしれない」「なかった」を含むもののみ
print("1回目調査でPLEなし\n", ple_neg)

ple_neg_oc = pd.merge(ple_neg, oc_2nd, left_index=True, right_index=True)  # 第１期にpleなしのうち、OCにNaNなし
print("第１期にpleなしのうち、OCにNaNなし\n", ple_neg_oc)
print("NaN個数\n", ple_neg_oc.isnull().sum(axis=0))  # 対象者に第1期でPLEが欠損値の人はいない

cols_to_use = first.columns.difference(ple_neg_oc.columns)
print("第１期量的データにあって、PLEやOCSに含まれない項目を検出\n", cols_to_use)
df4grf = ple_neg_oc.join([first[cols_to_use], base_1st, oc_naive, aq_2nd, ple_3rd, ple_4th], how='inner')
print("set_indexがなされているか？\n", df4grf)
df4grf.to_csv("test2.csv")
# 共変数から文字列を含む列、PLEの_2、baseに含まれる列を削除したもの

print("NaN個数\n", df4grf["OCS_0or1"].isnull().sum())
print("OCSあり\n", df4grf["OCS_0or1"].sum())
