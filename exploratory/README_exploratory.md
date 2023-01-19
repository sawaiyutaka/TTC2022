使う順番


・
1_dfmaker.py

　第1期のエクセルファイルを読み込み、データフレームに整形

｜

・
2_ocs_ple_merge.py

　第2期、第3期、第4期のエクセルファイルを読み込み、データフレームに整形

｜

・
3_nan_check.py

　第1期のデータの欠損値を確認
　第1期の欠損値の個数に応じて、項目を除外

｜

・
4_missforest.py

　第1期欠損値をmissforestを使って補完

｜

・
5b_binary.py

　ocsカットオフ以上の内、

　第3期pleで１項目でも2以上→1

　第3期pleで全て1→0

　として、group_3rdを設定

｜

・
5c_boruta_binary.py

　5bの0/1を弁別するrandomforestclassifierを作成
　borutaで変数選択

｜

・
5d_optuna.py

　5cに使うハイパーパラメータの最適化