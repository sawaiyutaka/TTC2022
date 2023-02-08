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

# 第２期のAQ素点を除外
df = df.drop(["BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132"], axis=1)


# 第２期の強迫を除外
df = df.drop(["BB39", "BB56", "BB57", "BB73", "BB83", "BB95", "BB96", "BB116", "OCS_sum"], axis=1)

# 第1期の強迫、PLEを除外
df = df.drop(["AB71", "AB87", "AB88", "AB104", "AB114", "AB126", "AB127", "AB145", "OCS_1st_sum"], axis=1)
df = df.drop(["AD57", "AD58", "AD59", "AD60", "AD61", "AD62", "AD63"], axis=1)

# 第1期いじめられを「この2か月間で、お子さんは、他の子からいじめられましたか？」のみに
# df = df.drop(["AD18", "AD19", "AD20", "AD21"], axis=1)

# 第1期の養育環境、養育態度を絞り込む
# df = df.drop(["AA56", "AA57", "AA59", "AA60", "AA61", "AB50", "AB51", "AB52", "AB53", "AB54", "AB56", "AB57"], axis=1)
print(df)

df.to_csv("test4.csv")