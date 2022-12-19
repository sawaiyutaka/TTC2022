import pandas as pd


df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_outcome/base_ple_imputed.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

# PLEの合計点を作成(第3期)
df_Y = df[["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"]]
print("df_Y\n", df_Y)
df["PLE_sum_3rd"] = df_Y.sum(axis=1)
print("第3回PLE合計\n", df["PLE_sum_3rd"])


# PLEの合計点を作成(第4期)
df_Y = df[["DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1", "DD69_1", "DD70_1", "DD71_1", "DD72_1"]]
print("df_Y\n", df_Y)
df["PLE_sum_4th"] = df_Y.sum(axis=1)
print("第4回PLE合計\n", df["PLE_sum_4th"])


# AQの合計点を作成
df_AQ = df[["BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132"]]
print(df_AQ)

print(df_AQ.replace({"BB123": {1: 0, 2: 0, 3: 1, 4: 1, 5: 0}}))

for i in ["BB123", "BB124", "BB128", "BB129", "BB130", "BB131"]:
    df_AQ = df_AQ.replace({i: {1: 0, 2: 0, 3: 1, 4: 1, 5: 0}})
    print(df_AQ)

for i in ["BB125", "BB126", "BB127", "BB132"]:
    df_AQ = df_AQ.replace({i: {1: 1, 2: 1, 3: 0, 4: 0, 5: 0}})
    print(df_AQ)

df["AQ_sum"] = df_AQ.sum(axis=1)
print("第2回AQ合計\n", df["AQ_sum"])
df.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/X_T_Y_imputed.csv")

# Xから回答日、回答時点の月齢は削除、調査員番号や調査員訪問日も削除
X = df.drop(["AA1YEAR", "AA1MONTH", "AA1DAY", "AA1age",
             "VS1", "VS2D", "VS2M", "VS2Y", "VS3", "VS4", "VS5,", "VS6", "VS7"], axis=1)

# 第3期のPLEを除外
X = X.drop(["PLE_sum_3rd", "CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"],
           axis=1)

# 第4期のPLEを除外
X = X.drop(["PLE_sum_4th",
            "DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1", "DD69_1", "DD70_1", "DD71_1", "DD72_1"], axis=1)

# 第２期の強迫を除外
X = X.drop(["BB39", "BB56", "BB57", "BB73", "BB83", "BB95", "BB96", "BB116", "OCS_sum", "OCS_0or1"], axis=1)

# 第２期のAQ素点を除外
X = X.drop(["BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132"], axis=1)

print(X)
# 第1期の強迫、PLEを除外
X = X.drop(["AB71", "AB87", "AB88", "AB104", "AB114", "AB126", "AB127", "AB145"], axis=1)
X = X.drop(["AD57", "AD58", "AD59", "AD60", "AD61", "AD62"], axis=1)

X = X.loc[:, ~X.columns.duplicated()]
print("重複を削除\n", X)
X.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/X_imputed.csv")
