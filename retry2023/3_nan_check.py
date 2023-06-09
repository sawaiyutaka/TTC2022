import pandas as pd
import seaborn as s
from matplotlib import pyplot as plt

data4grf = pd.read_table("/Volumes/Pegasus32R8/TTC/2023retry/data4grf_before_impute.csv",
                         delimiter=",", low_memory=False)
data4grf = data4grf.set_index("SAMPLENUMBER")
print(data4grf)

data4grf = data4grf.drop([
    "AB226", "AB227", "AB228", "AB229",  # 第二次性徴
    "AA185", "AA186", "AA187", "AA188",  # 両親アルコール
    "AD36", "AD37", "AD38", "AD39", "AD40", "AD41", "AD42", "AD43", "AD44", "AD45", "AD46", "AD47", "AD48",
    "AD49", "AD50", "AD51", "AD52", "AD53", "AD54", "AD55", "AD56",  # CPAQ
    "AC28", "AC28", "AC29", "AC30", "AC31", "AC32",  # 子WHO5
    "AA205", "AA206", "AA207", "AA208", "AA209", "AA210",  # 母K6
    "AB202", "AB203", "AB204", "AB205", "AB206", "AB207", "AB208",
    "AB209", "AB210", "AB211", "AB212", "AB213", "AB214", "AB215",  # 母PPL
    "AA213A", "AA213NIN", "AA213B", "AA214A", "AA214NIN", "AA214B", "AA215A", "AA215NIN",
    "AA215B", "AA216A", "AA216NIN", "AA216B", "AA217A", "AA217NIN", "AA217B", "AA218A", "AA218NIN", "AA218B",  # 母SSQ
    "AB35", "AB36", "AB37", "AB38", "AB39", "AB40", "AB41", "AB42", "AB43", "AB44",  # webaddiction
    "VS9", "VS10", "VS11", "VS12",  # SelfRegulation
    "AE6", "AE7", "AE8", "AE9",  # 2D4D
    "AC81", "AC82", "AC83", "AC84", "AC85", "AC86", "AC87", "AC88", "AC89",  # 子time discount
    "AB186", "AB187", "AB188", "AB189", "AB190", "AB191", "AB192", "AB193", "AB194"  # 母time discount
], axis=1)

col = len(data4grf)
NUM_0F_NAN = int(col * 0.05)  # 欠損値が何人未満の項目を使うか

print("各列のNaN個数\n", data4grf.isnull().sum())

# 第3期と第4期のPLEは欠損値の扱いが異なるため、分けておく
outcome = data4grf.filter(regex='^(C|D)', axis=1)
print(outcome)

# 欠損値がNUM OF NAN未満の項目のみ抽出
sr = data4grf.isnull().sum()
sr.to_csv("/Volumes/Pegasus32R8/TTC/2023retry/NaN_in_data4grf.csv")

s.set()
s.displot(sr)
sr = sr[sr < NUM_0F_NAN]

print(sr)
print(sr.index)  # 共変量の中でNaNが●個未満の項目名を抽出
name_columns = pd.DataFrame(sr, columns=["num_of_NaN"])
print("共変量の中でNaNが規定未満の項目\n", name_columns)

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
print("1期、2期のデータの内、NaNが5%未満の項目のみ抽出した\n", df2)
df2.to_csv("/Volumes/Pegasus32R8/TTC/2023retry/columns_NAN_under_5percent.csv")




