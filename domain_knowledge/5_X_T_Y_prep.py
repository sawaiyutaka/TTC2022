import pandas as pd
import codecs

df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022domain/data4grf_before_imp.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)


# 第1期いじめられを「この2か月間で、お子さんは、他の子からいじめられましたか？」のみに
# df = df.drop(["AD18", "AD19", "AD20", "AD21"], axis=1)

# 第1期の養育環境、養育態度を絞り込む
# df = df.drop(["AA56", "AA57", "AA59", "AA60", "AA61", "AB50", "AB51", "AB52", "AB53", "AB54", "AB56", "AB57"], axis=1)
print(df)

df.to_csv("/Volumes/Pegasus32R8/TTC/2022domain/x_t_y_before_imp.csv")
