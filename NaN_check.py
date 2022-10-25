import pandas as pd
import seaborn as s
from matplotlib import pyplot as plt


# 現在の最大表示列数の出力
# pd.get_option("display.max_columns")

# 最大表示列数の指定（ここでは50列を指定）
# pd.set_option('display.max_columns', 300)

all_1st = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv/TTC2022_1st_all.csv", delimiter=",", low_memory=False)
print(all_1st)

print("ple_naiveの中のNaN個数\n", all_1st.isnull().sum())
sr = all_1st.isnull().sum()
# sr.to_csv("NaN_in_all_1st.csv")

s.displot(sr)
plt.show()

# １期の量的データは全部で1477項目
sr = sr[sr < 500]  # 200人以上だと586項目, 150人以上だと622項目, 100人以上だと718項目
# sr = sr[sr > 250]  # 576
# sr = sr[sr > 500]  # 528
print(sr)
print(sr.index)  # NaNが●以上の項目名を抽出★

df = all_1st[sr.index]
print("columns under cutoff\n", df)
# df.to_csv("/Volumes/Pegasus32R8/TTC/2022csv/columns_NAN_under_500.csv")

name_columns = pd.DataFrame(sr, columns=["num_of_NaN"])
print(name_columns)

"""
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
"""
