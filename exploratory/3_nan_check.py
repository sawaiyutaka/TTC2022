import pandas as pd
import seaborn as s
from matplotlib import pyplot as plt

NUM_0F_NAN = int(1844 * 0.05)  # 欠損値が何人未満の項目を使うか

# 現在の最大表示列数の出力
# pd.get_option("display.max_columns")

# 最大表示列数の指定（ここでは50列を指定）
# pd.set_option('display.max_columns', 50)

data4grf = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_boruta/data4grf_before_impute.csv",
                         delimiter=",", low_memory=False)
data4grf = data4grf.set_index("SAMPLENUMBER")
print(data4grf)

print("NaN個数\n", data4grf.isnull().sum())

# 第3期と第4期のPLEは分けておく
outcome = data4grf.filter(regex='^(C|D)', axis=1)
print(outcome)

# 欠損値がNUM OF NAN未満の項目のみ抽出
sr = data4grf.isnull().sum()
sr.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_boruta/NaN_in_data4grf.csv")

s.set()
s.displot(sr)
# plt.show()

# １期の量的データは全部で1266項目(アウトカム指標を含み、重複を削除)
sr = sr[sr < NUM_0F_NAN]  # 5%(92人)未満だと765項目

print(sr)
print(sr.index)  # 共変量の中でNaNが●個未満の項目名を抽出★
name_columns = pd.DataFrame(sr, columns=["num_of_NaN"])
print(name_columns)

df1 = data4grf[sr.index]
print("columns under cutoff\n", df1)

# 参加者ごとの欠損値を表示
print("NaN個数\n", df1.isnull().sum(axis=1))
sr2 = df1.isnull().sum(axis=1) / len(df1.columns) * 100
s.set()
s.displot(sr2)
plt.show()

# アウトカムと結合
df2 = pd.merge(df1, outcome, left_index=True, right_index=True)
print(df2)
df2.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_boruta/columns_NAN_under_" + str(NUM_0F_NAN) + ".csv")




