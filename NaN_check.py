import pandas as pd
import seaborn as s
from matplotlib import pyplot as plt

NUM_0F_NAN = 150  # 欠損値が何人未満の項目を使うか

# 現在の最大表示列数の出力
# pd.get_option("display.max_columns")

# 最大表示列数の指定（ここでは50列を指定）
# pd.set_option('display.max_columns', 50)

all_1st = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_outcome/TTC2022_1st_outcome_Imp.csv",
                        delimiter=",", low_memory=False)
all_1st = all_1st.set_index("SAMPLENUMBER")
print(all_1st)

print("NaN個数\n", all_1st.isnull().sum())
sr = all_1st.isnull().sum()
# sr.to_csv("NaN_in_all_1st.csv")

s.displot(sr)
plt.show()

# １期の量的データは全部で1266項目(アウトカム指標を含み、重複を削除)
sr = sr[sr < NUM_0F_NAN]  # 100人未満だと580項目, 150人未満だと629項目, 200人未満で679項目

print(sr)
print(sr.index)  # NaNが●以上の項目名を抽出★

df = all_1st[sr.index]
print("columns under cutoff\n", df)
df.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/columns_NAN_under_" + str(NUM_0F_NAN) + ".csv")

name_columns = pd.DataFrame(sr, columns=["num_of_NaN"])
print(name_columns)

