import pandas as pd
import numpy as np
import glob
import codecs
import seaborn as s
from matplotlib import pyplot as plt

OCS_CUT_OFF = 12  # 強迫のCBCLカットライン。（-「項目数」点）以下は強迫なしとする。
# 2項目だと、1854人中290人がOCS 1となる

# 第１期基礎データ
with codecs.open("/Volumes/Pegasus32R8/TTC/2022base_OC_PLE/180511AB基本セット（CBCL・SDQ・SMFQ）_200720.csv", "r", "Shift-JIS",
                 "ignore") as file1:
    df1 = pd.read_table(file1, delimiter=",", low_memory=False)
    print(df1.head())
    df1 = df1[["SAMPLENUMBER", "TTC_sex", "AA1age", "AE1BMI",
               "AEIQ", "AAFedu", "AAMedu", "AA79Fsep", "AB161MIQ", "AA127Respondent"]]
    print(df1.head())

    base_1st = df1.replace(r'^\s+$', np.nan, regex=True)  # 空白をNaNに置換
    print("第１期基礎データ\n", base_1st)
    base_1st = base_1st.set_index("SAMPLENUMBER")

# プログラム2｜強迫、PLEデータを入れたフォルダ内の「data*.xlsx」を取得
files = glob.glob('/Volumes/Pegasus32R8/TTC/2022base_OC_PLE/*.xlsx')

# プログラム3｜変数listを空リストで設定
ls = []

# プログラム4｜プログラム2で取得したエクセルを一つずつpandasデータとして取得
for file in files:
    d = pd.read_excel(file)
    print(file)
    d = d.set_index("SAMPLENUMBER")
    # print(d)
    ls.append(d)

# プログラム5｜listに格納されたエクセルファイルをpandasとして結合
oc_ple = pd.concat(ls, axis=1, join='inner')
print(oc_ple)


# AQ
aq_2nd = oc_ple[["BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132"]]
print("第２期のAQ\n", aq_2nd)

# 第２期のOC
oc_2nd = oc_ple[["BB39", "BB56", "BB57", "BB73", "BB83", "BB95", "BB96", "BB116"]]  # 8項目ver
# oc_2nd = oc_ple[["BB39", "BB83"]]  # 2項目ver (2点以上は45人)
print("第２期のOC\n", oc_2nd)

# 第3期PLEのデータフレーム
ple = oc_ple.filter(like='_1', axis=1)
print("第3期PLEのうち、頻度を聞く項目のみ抽出：\n", ple)  # PLEのうち、頻度を聞く項目のみに絞りたい

ple_3rd = ple.filter(like='CD', axis=1)  # PLEのうち、第3期のみ
ple_3rd = ple_3rd.drop('CD70_1', axis=1)  # PLE第3期のうち、頻度以外の項目を削除
print("第3期PLE：\n", ple_3rd)

# 第4期PLEのデータフレーム
ple = oc_ple.filter(like='_1', axis=1)
print("第4期PLEのうち、頻度を聞く項目のみ抽出：\n", ple)  # PLEのうち、頻度を聞く項目のみに絞りたい

ple_4th = ple.filter(like='DD', axis=1)  # PLEのうち、第4期のみ
ple_4th = ple_4th.drop('DD77_1', axis=1)  # PLE第4期のうち、頻度以外の項目を削除
print("第4期PLE：\n", ple_4th)


# 第１期にPLEがなかった群を抽出
df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022domain/TTC2022_1st_all.csv",
                   delimiter=",", low_memory=False)
df = df.set_index("SAMPLENUMBER")
print(df.head())
# print("第１期データのNaN個数\n", df.isnull().sum())

# 1回目調査でPLEなしを抽出したい
df_PLE = df[["AD57", "AD58", "AD59", "AD60", "AD61", "AD62", "AD63"]]
print("df_PLE\n", df_PLE)
print("NaN個数\n", df_PLE.isnull().sum(axis=0))
ple_1st_pos = (df_PLE == 1.0)
print("Yes, definitelyの回答をTRUEに置換\n", ple_1st_pos)
print("1回目調査でPLEが「Yes, definitely」\n", ple_1st_pos.sum())
print("ple_1st_pos\n", df_PLE[ple_1st_pos])
ple_pos = df_PLE[ple_1st_pos].dropna(how='all')
print("1回目調査でPLEがYes. definitelyが１つでもある\n", ple_pos)
# ple_negと排反になることを確認

ple_1st_neg = ((df_PLE == 3.0) | (df_PLE == 2.0))
print("PLE 2か3をTRUEに", ple_1st_neg)
print("1回目調査でPLEなしの人数\n", ple_1st_neg.sum())
print("ple_1st_neg\n", df_PLE[ple_1st_neg])
ple_neg = df_PLE[ple_1st_neg].dropna(how='any')  # ★第１期に「あったかもしれない」「なかった」を含むもののみ
print("1回目調査でPLEなし\n", ple_neg)
print("NaN個数\n", ple_neg.isnull().sum(axis=0))  # 第1期でPLEが欠損値の人

cols_to_use = df.columns.difference(ple_neg.columns)
print("第１期量的データにあって、PLEやOCSに含まれない項目を検出\n", cols_to_use)
df4nan_check = ple_neg.join([df[cols_to_use], base_1st, aq_2nd, oc_2nd, ple_3rd, ple_4th], how='inner')
print("set_indexがなされているか？\n", df4nan_check)

# 回答日、回答時点の月齢は削除、調査員番号や調査員訪問日も削除
data4nan_check = df4nan_check.drop(["AA1YEAR", "AA1MONTH", "AA1DAY", "AA1age",
                        "VS1", "VS2D", "VS2M", "VS2Y", "VS3", "VS4", "VS5", "VS6", "VS7"], axis=1)

print("回答日、回答時点の月齢は削除、調査員番号や調査員訪問日を削除\n", data4nan_check)
data4nan_check.to_csv("/Volumes/Pegasus32R8/TTC/2022domain/data4nan_check.csv")
# 共変数から文字列を含む列、PLEの_2、baseに含まれる列を削除したもの
