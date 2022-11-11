import pandas as pd
import seaborn as s
from matplotlib import pyplot as plt

NUM_0F_NAN = 150  # 欠損値が何人未満の項目を使うか

all_1st = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_outcome/TTC2022_1st_outcome.csv", delimiter=",", low_memory=False)
print(all_1st)

print("NaN個数\n", all_1st.isnull().sum())
sr = all_1st.isnull().sum()
# sr.to_csv("NaN_in_all_1st.csv")

s.displot(sr)
plt.show()

# １期の量的データは全部で1446項目(アウトカム指標を含み、重複を削除)
sr = sr[sr < NUM_0F_NAN]  # 50人未満だと527項目, 100人未満だと684項目, 150人未満だと743項目
# sr = sr[sr < 200]  # 200人未満で799項目
# sr = sr[sr < 500]  # 500人未満で904項目
print(sr)
print(sr.index)  # NaNが●以上の項目名を抽出★

df = all_1st[sr.index]
print("columns under cutoff\n", df)
df.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/columns_NAN_under_150.csv")

name_columns = pd.DataFrame(sr, columns=["num_of_NaN"])
print(name_columns)

