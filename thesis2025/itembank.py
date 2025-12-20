import pandas as pd

# --- 1. CSVファイルの列名を取得 ---
csv_file = "/Volumes/Pegasus32R8/TTC/2025thesis/df4boruta.csv"
df_csv = pd.read_csv(csv_file)
col_names = df_csv.columns.tolist()   # 列名をリストとして取得
print("CSV列名:", col_names)

# --- 2. Excelファイルの「データ番号」と照合して行を抽出 ---
excel_file = "/Volumes/Pegasus32R8/TTC/200715_TTC_itembank_labelling.xlsx"
df_excel = pd.read_excel(excel_file)

# Excelの「データ番号」列がCSV列名に含まれる行だけ抽出
matched_rows = df_excel[df_excel["データ番号"].isin(col_names)]

selected_columns = ["データ番号", "ドメイン", "項目", "選択肢", "引用元and/or初出"]
matched_rows = matched_rows[selected_columns]

# 結果を確認
print(matched_rows)

# 必要なら保存
matched_rows.to_excel("/Volumes/Pegasus32R8/TTC/2025thesis/matched_rows.xlsx", index=False)
