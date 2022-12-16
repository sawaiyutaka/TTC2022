使う順番

1 main.py

　第1期のエクセルファイルを読み込み、データフレームに整形


2 NaN_check.py

　第1期のデータの欠損値を確認
　第1期の欠損値の個数に応じて、項目を除外


3 OC_PLE_prep.py

　第2期、第3期、第4期のエクセルファイルを読み込み、データフレームに整形


4 impute_nan.py

　第1期および第3期・第4期のPLE欠損値をランダムフォレストを使って補完

5 X_T_Y_prep.py

　PLEやAQ10など素点から計算


6 GRF_3rd.py, GRF_4th.py(順不同)

　一般化ランダムフォレストによるtreatment effectの計算

（第2期で強迫症状がない場合と、強迫症状がある場合で、PLEのスコアがどれだけ異なるか推定）
 
　treatment effectの予測に関わった共変量（第1期のデータ）を示す：shap valueを使用



(中止)shap_of_cate.py

　treatment effectと第1期データを結合して、

　treatment effectの予測に関わった共変量（第1期のデータ）を示す：shap valueを使用


(保留)predictor_random4TTC
