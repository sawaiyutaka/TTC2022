import pandas as pd

# 現在の最大表示列数の出力
# pd.get_option("display.max_columns")

# 最大表示列数の指定（ここでは50列を指定）
# pd.set_option('display.max_columns', 300)

df = pd.read_table("TTC2022_base_minimum.csv", delimiter=",")
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
OCS_CUT_OFF = 12  # -8点がスコア
df_OCS_pos = (df_OCS["OCS_sum"] > OCS_CUT_OFF) # 5点以上だと115人
print("df_OCS_pos\n", df_OCS_pos)
print("強迫症状カットオフ以上\n", df_OCS_pos.sum())


print("df_OCSカットオフ以上以上\n", pd.merge(df_PLE, df_OCS[df_OCS_pos], left_index=True, right_index=True)) # df_OCSで13点以上を抽出
df0 = pd.merge(df_PLE[df_PLE_neg], df_OCS, left_index=True, right_index=True) # かつ、PLEなし
print("PLEなし\n", df0.dropna(how='any'))  # NaNを削除
df0 = df0.dropna(how='any')
df1 = pd.merge(df_PLE[df_PLE_neg], df_OCS[df_OCS_pos], left_index=True, right_index=True) # df_OCS13以上でかつPLEなし
print("df_OCSカットオフ以上でかつPLEなし\n", df1.dropna(how='any'))  # NaNを削除


# OCS13以上を1、12以下を0にする
df_OCS["OCS_0or1"] = (df_OCS["OCS_sum"] > OCS_CUT_OFF)*1
print(df_OCS)  # 2733行

cols_to_use = df.columns.difference(df0.columns)
df_grf = df0.join([df[cols_to_use], df_OCS["OCS_0or1"]], how='inner')
print(df_grf)
# df_grf.to_csv("TTC2022_ple_naive.csv")  # 共変数から文字列を含む列、PLEの_2、baseに含まれる列を削除したもの

print("NaN個数\n", df_grf.isnull().sum())
# s=df_grf.isnull().sum()
# print(s.head(264))
# s.to_csv("TTC2022_grf_min_NaN.csv")

# 共変数から文字列を含む列、PLEの_2、baseに含まれる列を削除したもの
# 1期でPLEなし
df_pre_imputed = pd.read_table("TTC2022_ple_naive.csv", delimiter=",")
print(df_pre_imputed.describe())
print(df_pre_imputed.head())
df_pre_imputed = df_pre_imputed.set_index("SAMPLENUMBER")
print(df_pre_imputed.head())

# 列ごとにNaNをいくつ含むか
print("ple_naiveの中のNaN個数\n", df_pre_imputed.isnull().sum())
s = df_pre_imputed.isnull().sum()
# s.to_csv("NaN_in_ple_naive.csv")
