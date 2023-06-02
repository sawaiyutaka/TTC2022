import pandas as pd


d = pd.read_excel("/Volumes/Pegasus32R8/TTC/2022rawdata_copy/171114A子ペット.xlsx", sheet_name='作業')
print(d)
d = d.set_index("SAMPLENUMBER")
d0 = d[["AE10"]]
print(d0)
d1 = d.filter(regex='Kind$', axis=1)  # Impで終わる列＝欠損値１以下なら平均値で補完
print(d1)
d2 = 1-d1.isna()*1
print(d2)
d2.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/pet_kind.csv")

d3 = pd.merge(d0, d2, left_index=True, right_index=True)
print(d3)
print(d3["AE10"].isnull().sum())