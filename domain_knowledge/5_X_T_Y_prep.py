import pandas as pd
import codecs

df = pd.read_table("test3.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

# 第２期のAQ素点からAQを計算
# AQの合計点を作成
df_AQ = df[["BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132"]]
print(df_AQ)

for i in ["BB123", "BB124", "BB128", "BB129", "BB130", "BB131"]:
    df_AQ = df_AQ.replace({i: {1: 0, 2: 0, 3: 1, 4: 1, 5: 0}})
    print(df_AQ)

for i in ["BB125", "BB126", "BB127", "BB132"]:
    df_AQ = df_AQ.replace({i: {1: 1, 2: 1, 3: 0, 4: 0, 5: 0}})
    print(df_AQ)

df["AQ_sum"] = df_AQ.sum(axis=1)
print("第2回AQ合計\n", df["AQ_sum"])

print(df)

# 第２期のAQ素点を除外
df = df.drop(["BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132"], axis=1)

# チック全項目を統一



df.to_csv("test4.csv")

"""
# 第3期のPLEを除外
X = df.drop(["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"],
            axis=1)

# 第4期のPLEを除外
X = X.drop(["DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1", "DD69_1", "DD70_1", "DD71_1", "DD72_1"], axis=1)

# 第２期の強迫を除外
X = X.drop(["BB39", "BB56", "BB57", "BB73", "BB83", "BB95", "BB96", "BB116", "OCS_sum", "OCS_0or1"], axis=1)


print(X)

# 第1期の強迫、PLEを除外
X = X.drop(["AB71", "AB87", "AB88", "AB104", "AB114", "AB126", "AB127", "AB145"], axis=1)
X = X.drop(["AD57", "AD58", "AD59", "AD60", "AD61", "AD62"], axis=1)

X = X.loc[:, ~X.columns.duplicated()]
print("重複を削除\n", X)
X.to_csv("test5.csv")
"""