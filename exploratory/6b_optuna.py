# 目的関数の定義（最小値問題として定式化する。）
from multiprocessing import cpu_count

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate
import optuna


def objective(trial):
    # min_samples_split = trial.suggest_int("min_samples_split", 8, 16)
    # max_leaf_nodes = int(trial.suggest_discrete_uniform("max_leaf_nodes", 4, 64, 4))
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    n_estimators = trial.suggest_int('n_estimators', 100, 2000, 50)
    max_depth = trial.suggest_int('max_depth', 2, 7)  # , log=True)
    max_features = trial.suggest_categorical('max_features', [1.0, 'sqrt', 'log2'])

    clf = RandomForestClassifier(
        max_depth=max_depth,
        max_features=max_features,
        n_estimators=n_estimators,
        # min_samples_split=min_samples_split,
        # max_leaf_nodes=max_leaf_nodes,
        criterion=criterion,
        random_state=0,
        n_jobs=int(cpu_count() * 2 / 3))

    clf.fit(X_train, Y_train)

    skf = StratifiedKFold(n_splits=10,
                          shuffle=True,
                          random_state=0)
    scoring = {'accuracy': make_scorer(accuracy_score)}
    scores_skf = cross_validate(clf,
                                X,  #
                                y,
                                cv=skf,
                                n_jobs=int(cpu_count() / 2),
                                scoring=scoring)
    score = scores_skf['test_accuracy'].mean()

    return 1.0 - score  # accuracy_score(Y_test, clf.predict(X_test))


df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_boruta/binary.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

y = df["group"]
print(y)

X = df.drop(["group"], axis=1)

# 参照！！：https://note.com/utaka233/n/ne71851e1d678

print(X.shape)
print(X.head())

Y_train, Y_test, X_train, X_test = train_test_split(y, X, test_size=0.2, stratify=y, random_state=0)
print("Y_train", Y_train)
print("Y_test", Y_test)

# ハイパーパラメータの自動最適化
study = optuna.create_study()
study.optimize(objective, n_trials=2000)

print(study.best_params)  # 求めたハイパーパラメータ
print("正答率: ", 1.0 - study.best_value)
