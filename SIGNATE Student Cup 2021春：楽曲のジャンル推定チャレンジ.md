# はじめに

書籍でも触れたことは無かった為、マルチクラス分類に挑戦したいと考えコンペを調べていた所、[SIGNATE Student Cup 2021春：楽曲のジャンル推定チャレンジ](https://signate.jp/competitions/565)存在を知りチャレンジ。
結果は2024年12月5日時点で635人中**129位**でした。

![スクリーンショット 2024-12-05 234419.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3794471/d5d8c211-0fff-00bb-afbb-ce5a41be386e.png)

# 実行環境
| カテゴリー | 名称 | バージョン　|
|:-:|:-:|:-:|
|パソコン|Surface Pro 6|10.0.19045 ビルド 19045　|
|開発環境|Visual Studio Code|1.94.2 (user setup)　|
|言語|Python|3.9.1|
|ライブラリ|scikit-learn|1.4.2|
|ライブラリ|lightgbm|4.3.0|
|ライブラリ|xgboost|2.1.1|

# 目次
[1. データの確認](#データの確認) 

[2. kNNでのモデル構築](#kNNでのモデル構築) 

[3. アンサンブル](#アンサンブル)

[4. その他試した事](#その他試した事)

[5. まとめ](#まとめ)

# データの確認
**ライブラリのインポート**
```python
import pandas as pd
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sweetviz as sv

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold

import lightgbm as lgb
import xgboost as xgb
```

**データの読み込みと構造の確認**

```python
# パスの指定
path_train = os.path.join(r"Data\train.csv")
path_test = os.path.join(r"Data\test.csv")
path_genre_labels = os.path.join(r"Data\genre_labels.csv")
path_sample = os.path.join(r"Data\sample_submit.csv")

# データの読み込み
train_df = pd.read_csv(path_train)
test_df = pd.read_csv(path_test)
genre_labels_df = pd.read_csv(path_genre_labels)
sample_df = pd.read_csv(path_sample, names=["index", "genre"])

# データ構造の確認
print(f"Train_df.shape : {train_df.shape}")
print(f"test_df.shape : {test_df.shape}")
```
```python
Train_df.shape : (4046, 14)
test_df.shape : (4046, 13)
```

**欠損値とデータ型の確認**

```python
# 欠損値とデータ型確認関数の定義
def check_null(tr_df, va_df):
    # train_dataの欠損値格納データフレームの作成
    train_info_df = pd.DataFrame(columns=["col", "Null_tr", "Dtype_tr"])
    # 各列の欠損値とデータ型をデータフレームに格納
    for col in tr_df.columns:
        col_df = pd.DataFrame({
            "col" : [col],
            "Null_tr" : tr_df[col].isna().sum(),
            "Dtype_tr" : tr_df[col].dtype
        })
        train_info_df = pd.concat([train_info_df, col_df], axis=0)

    # test_dataの欠損値格納データフレームの作成
    test_info_df = pd.DataFrame(columns=["col", "Null_te", "Dtype_te"])
    # 各列の欠損値とデータ型をデータフレームに格納
    for col in va_df.columns:
        col_df = pd.DataFrame({
            "col" : [col],
            "Null_te" : va_df[col].isna().sum(),
            "Dtype_te" : va_df[col].dtype
        })
        test_info_df = pd.concat([test_info_df, col_df], axis=0)
    # train_dataとtest_dataのデータフレームを結合
    info_df = pd.merge(train_info_df, test_info_df, on="col", how="left")

    return info_df

check_null(train_df, test_df)
```
|    | col              |   Null_tr | Dtype_tr   |   Null_te | Dtype_te   |
|---:|:-----------------|----------:|:-----------|----------:|:-----------|
|  0 | index            |         0 | int64      |         0 | int64      |
|  1 | genre            |         0 | int64      |       nan | nan        |
|  2 | popularity       |         0 | int64      |         0 | int64      |
|  3 | duration_ms      |         0 | int64      |         0 | int64      |
|  4 | acousticness     |         0 | float64    |         1 | float64    |
|  5 | positiveness     |        10 | float64    |        14 | float64    |
|  6 | danceability     |         8 | float64    |        11 | float64    |
|  7 | loudness         |         0 | float64    |         0 | float64    |
|  8 | energy           |         0 | float64    |         1 | float64    |
|  9 | liveness         |         3 | float64    |         6 | float64    |
| 10 | speechiness      |         8 | float64    |        11 | float64    |
| 11 | instrumentalness |         1 | float64    |         2 | float64    |
| 12 | tempo            |         0 | object     |         0 | object     |
| 13 | region           |         0 | object     |         0 | object     |

**相関係数の確認**
```python
# 数値列のみで相関行列を計算
cor = train_df.iloc[:, :12].corr()
# 図の表示サイズを指定
plt.figure(figsize=(8,8))
# 描画
sns.heatmap(cor, cmap= sns.color_palette('coolwarm', 10), annot=True,fmt='.2f', vmin = -1, vmax = 1)
```
![output.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3794471/ed405ad5-21ba-622e-637b-756548c65df9.png)

**データの概要を確認**
```python
# sweetvizの実行
report = sv.compare(train_df, test_df, target_feat="genre")
# 結果をhtml形式で保存
report.show_html("report.html")
```

**関数定義**
```python
# merged_df作成関数
def make_merged_df(train_df, test_df):
    train_df["train"] = 1
    test_df["train"] = 0
    test_df["genre"] = -1
    merged_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    return merged_df

# merged_dfを再度train_df, test_dfに分割関数
def devide_merged_df(merged_df):
    train_df = merged_df[merged_df["train"] == 1].copy().drop(columns="train").reset_index(drop=True)
    test_df = merged_df[merged_df["train"] == 0].copy().drop(columns=["train", "genre"]).reset_index(drop=True)
    return train_df, test_df
# merged_dfの作成
merged_df = make_merged_df(train_df, test_df)
```
# kNNでのモデル構築
[フォーラム](https://signate.jp/competitions/449/discussions/knn-baseline-cv06256-lb06417)を参考

```python
## 特徴量エンジニアリング
# ジャンル名を追加
genre_dict = genre_labels_df.set_index("labels")["genre"].to_dict()
# mapでジャンル名を取得し、存在しない値があった場合は「Unknown」を返す
merged_df["genre_name"] = merged_df["genre"].map(genre_dict).fillna("Unknown")

# tempo列は最大と最小の平均値に変換
merged_df["tempo"] = merged_df["tempo"].map(lambda x: sum(map(int, x.split("-"))) / 2)

# regionをone-hotエンコーディング
merged_df = merged_df.join(pd.get_dummies(merged_df["region"])).rename(columns={"unknown" : "region_unknown"})

# 各行の欠損値の合計数の特徴量
merged_df["num_nans"] = 0
for col in [
    "acousticness",
    "positiveness",
    "danceability",
    "energy",
    "liveness",
    "speechiness",
    "instrumentalness",
]:
    merged_df["num_nans"] += merged_df[col].isna()

## feature scaling

# tempoのlog
merged_df["log_tempo"] = np.log(merged_df["tempo"])
# 標準化処理
for col in [
    'popularity', 'duration_ms', 'acousticness',
    'positiveness', 'danceability', 'loudness', 'energy', 'liveness',
    'speechiness', 'instrumentalness', 'log_tempo', 'num_nans',
]:
    merged_df["standardscaled_" + col] = StandardScaler().fit_transform(merged_df[[col]])[:, 0]

train_df, test_df = devide_merged_df(merged_df)

# 目的変数を変数に代入
target = train_df["genre"]
```
```python
# one-hotエンコーディングした地域と、スケーリングした特徴量を変数化
features = [col for col in train_df.columns if col.startswith("region_") or col.startswith("standard")]

# 空の辞書を作成
dict_feature_weights = {}

# regionの重みを設定
for col in train_df.columns:
    if col.startswith("region_"):
        dict_feature_weights[col] = 100.0

# 楽曲成分特徴量の重み設定
for col in train_df.columns:
    if col.startswith("standard"):
        dict_feature_weights[col] = 1.0

# 別途重みを設定
dict_feature_weights["standardscaled_popularity"] = 8.0
dict_feature_weights["standardscaled_log_tempo"] = 0.001
dict_feature_weights["standardscaled_num_nans"] = 100.0

# 重みを配列に
feature_weights = np.array([dict_feature_weights[col] for col in features])
```
```python
# kNNのparametersを設定
N_CLASSES = 11 
n_neighbors = 6 

# モデルの構築
model = KNeighborsClassifier(n_neighbors=n_neighbors + 1, weights="distance")

# 特徴量を欠損値0埋めし、Numpy配列に変換後、対応する重みを掛けている
X = train_df[features].fillna(0.0).values * feature_weights

# モデル学習
model.fit(X, target)

# 距離とインデックス配列を取得
distances, indexes = model.kneighbors(X)
# 配列から自分の距離(0)を取り除く
distances = distances[:, 1:]
indexes = indexes[:, 1:]
# 近傍6ポイントのgenre配列の作成
labels = target.values[indexes]
# train_dfの予測
preds = np.array([np.bincount(labels_, distances_, N_CLASSES) for labels_, distances_ in zip(labels, 1.0/distances)]).argmax(1)

# F1スコアの計算
score = f1_score(target, preds, average="macro")
print(f"f1_score = {score}")
print(classification_report(target, preds))

def visualize_confusion_matrix(y_true, pred_label, height=0.6, labels=None):
    # 混同行列を計算
    conf = confusion_matrix(y_true, pred_label, normalize='true')
    
    # ヒートマップのサイズを設定
    size = len(conf) * height
    fig, ax = plt.subplots(figsize=(size * 2, size * 1.5))
    
    # ヒートマップを描画
    sns.heatmap(conf, cmap='YlOrBr', ax=ax, annot=True, fmt='.2f')
    ax.set_ylabel('Label')
    ax.set_xlabel('Predict')
    
    # ラベルを設定
    if labels is not None:
        ax.set_yticklabels(labels, rotation=0)
        ax.set_xticklabels(labels, rotation=90)

    plt.show()  # 図を表示

# 関数の呼び出し
visualize_confusion_matrix(target, preds, labels=genre_labels_df["genre"])

# テストデータを予測
# モデル構築
model = KNeighborsClassifier(n_neighbors)
# データセットの作成
X_train = train_df[features].fillna(0.0).values * feature_weights
X_test = test_df[features].fillna(0.0).values * feature_weights
# モデル学習
model.fit(X_train, target)
# 予測
test_df["genre"] = model.predict(X_test)
df_submission = test_df[["index", "genre"]]
```
### F1スコア: 0.6524753599245728

| クラス | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.86      | 0.59   | 0.70     | 32      |
| 1     | 0.56      | 0.37   | 0.44     | 205     |
| 2     | 0.67      | 0.60   | 0.63     | 191     |
| 3     | 0.82      | 0.71   | 0.76     | 362     |
| 4     | 0.69      | 0.64   | 0.67     | 45      |
| 5     | 0.67      | 0.49   | 0.57     | 126     |
| 6     | 0.68      | 0.34   | 0.45     | 50      |
| 7     | 0.63      | 0.58   | 0.60     | 334     |
| 8     | 0.70      | 0.79   | 0.75     | 1305    |
| 9     | 0.75      | 0.88   | 0.81     | 59      |
| 10    | 0.77      | 0.81   | 0.79     | 1337    |

| 指標        | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Accuracy    |           |        | 0.72     | 4046    |
| Macro Avg   | 0.71      | 0.62   | 0.65     | 4046    |
| Weighted Avg| 0.72      | 0.72   | 0.72     | 4046    |

![output.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3794471/2ca7ff42-3d3e-eb05-cdfb-6c2474cb876b.png)


# アンサンブル
```python
# モデル学習～検証データ、テストデータ予測関数の作成
def train_and_predict(model, x_tr, y_tr, x_va, y_va, test):
    model.fit(x_tr, y_tr) # モデル学習
    valid_preds = model.predict(x_va)  # 検証データのクラスラベル予測
    valid_proba = model.predict_proba(x_va)  # 検証データのクラス確率予測
    f1_macro = f1_score(y_va, valid_preds, average='macro') # f1_macroを計算
    test_proba = model.predict_proba(test)  # テストデータのクラス確率予測
    return valid_preds, valid_proba, f1_macro, test_proba

# 学習と評価、予測を行う
def get_models_trained(train, test, target, numfolds):
    kf = StratifiedKFold(n_splits=numfolds, shuffle=True, random_state=42) # cvの設定
    oof_predictions = np.zeros(len(train), dtype=int) # oofの予測値格納配列の初期化 / 指定しないとfloat型になる
    test_predictions = np.zeros((len(test), len(np.unique(target)))) # 予測値の格納配列の初期化

    # モデルリストの作成
    models = [
        ('model1', lgb.LGBMClassifier(boosting_type = "gbdt", objective = "multiclass", metric = "multi_logloss", num_class= 11, random_state=123)),
        ('model2', lgb.LGBMClassifier(boosting_type = "gbdt", objective = "multiclass", metric = "multi_logloss", num_class= 11, random_state=42)),
        ('model3', xgb.XGBClassifier(objective= "multi:softmax", num_class= 11, eval_metric= "mlogloss", seed=123)),
        ('model4', xgb.XGBClassifier(objective= "multi:softmax", num_class= 11, eval_metric= "mlogloss", seed=42))
    ]

    for fold, (train_index, valid_index) in enumerate(kf.split(train, target)):
        # 学習データと検証データの分割
        x_tr, x_va = train.iloc[train_index], train.iloc[valid_index]
        y_tr, y_va = target.iloc[train_index], target.iloc[valid_index]
        # 各モデルの予測ラベル、F1スコア、テストデータ予測確率を格納するための変数を初期化
        valid_pred_dict = {}
        f1_dict = {}
        output_predictions = []
        # 各モデルで学習～予測までを実施
        for name, model in models:
            valid_preds, valid_proba, f1_macro, test_proba = train_and_predict(model, x_tr, y_tr, x_va, y_va, test)
            valid_pred_dict[name] = valid_preds # 検証データの予測値model名をキーにして格納
            f1_dict[name] = f1_macro # 検証データのf1_macroスコアをmodel名をキーにして格納
            output_predictions.append(test_proba) # モデル毎のテストデータの予測確率を格納
            # 1配列に0~11までの確率が格納され、それがtest_dfの長さ分(4096行)格納されている

        # 各foldの最頻値を計算
        valid_preds_mode = mode(np.array(list(valid_pred_dict.values())), axis=0)[0].flatten().astype(int)
        # 最頻値を用いてf1_macroを計算
        f1_macro_mode = f1_score(y_va, valid_preds_mode, average='macro')

        best_model_name = max(f1_dict, key=f1_dict.get) # f1_macroスコアが一番高いモデルを選定
        best_model = dict(models)[best_model_name] # モデルリストからベストモデルを抽出
        valid_preds_best_model = best_model.predict(x_va) # ベストモデルで検証データの予測
        f1_best = f1_score(y_va, valid_preds_best_model, average='macro') # f1_macroを計算

        # 最頻値でのf1スコアがベストモデルより高い場合
        if f1_macro_mode > f1_best:
            print(f"Fold {fold + 1}: Averaging all models. F1_macro mode: {f1_macro_mode}, Best F1_macro: {f1_best}")
            test_predictions += np.mean(output_predictions, axis=0) / kf.n_splits # 確率をfold数で割っている
            oof_predictions[valid_index] = valid_preds_mode # oofの予測値を更新
        # ベストモデルが最頻値でのf1スコアより高い場合
        else:
            print(f"Fold {fold + 1}: Using best model. F1_macro mode: {f1_macro_mode}, Best F1_macro: {f1_best}")
            test_predictions += test_proba / kf.n_splits
            oof_predictions[valid_index] = valid_preds_best_model

        print('---------------\n')

    # OOF F1_macroスコアの計算
    overall_f1_macro = f1_score(target, oof_predictions, average='macro')
    print(f"OOF F1_macro = {overall_f1_macro}")

    # 積み上げた予測確率を基に、test_dfの予測値を確定
    test_predictions = np.argmax(test_predictions, axis=1)

    return oof_predictions, test_predictions
```

```python
# knn予測で疑似ラベルを作成
pseudo_train_df = pd.concat([train_df, test_df], axis=0).reset_index()
# データセットの作成
pseudo_x_train = pseudo_train_df[features]
pseudo_y_train = pseudo_train_df["genre"]
pseudo_id_train = pseudo_train_df[["index"]]
pseudo_x_test = test_df[features]
pseudo_id_test = test_df[["index"]]
# 予測
off_preds, test_preds = get_models_trained(pseudo_x_train, pseudo_x_test, pseudo_y_train, 15)
# sample_submitを読み込み
sample_submit = pd.read_csv(r"Data\sample_submit.csv", names=["id", "Pred"])
# submit用dfの作成
ensemble_df = pd.DataFrame({
    "id" : sample_submit["id"],
    "pred" : test_preds
})
# csv出力
ensemble_df.to_csv("ensemble_bestlgb_lgb_xgb_xgb_cv15.csv", index=False, header=False) # 
```
```
# 出力結果
Fold 1: Averaging all models. F1_macro mean: 0.6884143278346596, Best F1_macro: 0.6774207252547452
Fold 2: Averaging all models. F1_macro mean: 0.6493209125006802, Best F1_macro: 0.6462297637199079
Fold 3: Using best model. F1_macro mean: 0.6389674141738765, Best F1_macro: 0.6495472468419812
Fold 4: Averaging all models. F1_macro mean: 0.742216696426971, Best F1_macro: 0.714318260268903
Fold 5: Averaging all models. F1_macro mean: 0.6934621322807956, Best F1_macro: 0.6877940185323589
Fold 6: Averaging all models. F1_macro mean: 0.7041493955669824, Best F1_macro: 0.7023699721051804
Fold 7: Using best model. F1_macro mean: 0.6971054049648324, Best F1_macro: 0.6991290720928357
Fold 8: Using best model. F1_macro mean: 0.6408778407227529, Best F1_macro: 0.6708528923467385
Fold 9: Using best model. F1_macro mean: 0.7322920574256524, Best F1_macro: 0.7445739758544566
Fold 10: Averaging all models. F1_macro mean: 0.7314280434477247, Best F1_macro: 0.7108116922423577
Fold 11: Averaging all models. F1_macro mean: 0.7191983109178124, Best F1_macro: 0.6991074925011225
Fold 12: Averaging all models. F1_macro mean: 0.6822294693002479, Best F1_macro: 0.6821401863030636
Fold 13: Averaging all models. F1_macro mean: 0.7249457009391257, Best F1_macro: 0.7219231139537208
Fold 14: Averaging all models. F1_macro mean: 0.6920002305971397, Best F1_macro: 0.6829328660437706
Fold 15: Averaging all models. F1_macro mean: 0.7227161626570857, Best F1_macro: 0.7095235273227455

OOF F1_macro = 0.7048474636058487
```

# その他試した事
<details>
<summary>折り畳み</summary>

**・特徴量作成、各モデルoptunaでのパラメータチューニング**

**・特徴量の分布と他特徴量との相関係数を確認の上、欠損値補完**
<details>
<summary>コード例</summary>

```python
def plot_data(tr_df, va_df, col_name):
    dfs = [tr_df, va_df]
    fig, axs = plt.subplots(ncols=2, figsize=(30, 15))
    
    for i, df in enumerate(dfs):
        # 平均値と中央値を計算
        mean_dance = df[col_name].mean()
        median_dance = df[col_name].median()

        # 分布のプロット
        sns.histplot(df[col_name], kde=True, label=col_name, color="#2ecc71", ax=axs[i])

        # 平均値と中央値の縦線を追加
        axs[i].axvline(mean_dance, color="r", linestyle="--", linewidth=2,label=f"{col_name}_mean: {mean_dance:.2f}")
        axs[i].axvline(median_dance, color="b", linestyle="--", linewidth=2, label=f"{col_name}_median: {median_dance:.2f}")

        # グラフのタイトルをデータフレーム名に設定
        axs[i].set_title(df.name, fontsize=35)
        
        # X軸とY軸のラベルの設定
        axs[i].set_xlabel(col_name, fontsize=25)
        axs[i].set_ylabel("Frequency", fontsize=15)

        # 目盛ラベルの設定
        axs[i].tick_params(axis="both", labelsize=20)
        
        # ラベルの設定
        axs[i].legend(fontsize=25)
    # レイアウトの自動調整
    plt.tight_layout()
    plt.show()

# positivenessと他の特徴量の相関係数の確認
train_df__corr = train_df.corr(numeric_only=True).unstack().sort_values(kind="quicksort", ascending=False).reset_index()
# 列名の変更
train_df__corr.rename(columns={"level_0": "Feature1", "level_1": "Feature2", 0: "Correlation Coefficient"}, inplace=True)

# danceabilityの相関関係を確認
train_df__corr[train_df__corr["Feature1"] == "danceability"]

# データ分布の確認
plot_data(train_df, test_df, "danceability")

# train/test共に分布も綺麗で、欠損値の数も8個と全体の約0.2%と影響は小さいため、平均値で補間を実施
train_df["danceability"] = train_df["danceability"].fillna(train_df["danceability"].mean()) # 平均値で欠損値を補間
test_df["danceability"] = test_df["danceability"].fillna(test_df["danceability"].mean()) # 平均値で欠損値を補間
```
そのほかの特徴量についても同様の処理を行った

</details>


**・オーバーサンプリング(SMOTE)、アンダーサンプリング(RandomUnderSampler)**

**・SVC, XGB, RFをベースモデルに、LogisticRegressionをメタモデルとしたスタッキング**

**・UMAPによる次元削減**

**・クラス結合(目的変数"1"と"8"を"18"とする等)してクラス数を減らし、複数回予測**

<details><summary>コード例</summary>

```python
# merged_dfを作成
merged_df = make_merged_df(train_df, test_df)
# 結合1
A, B = 1, 8
C = A * 10 + B
# 結合2
D, E = 2, 6
F = D * 10 + E
# 結合3
G, H = 4, 5
I = G * 10 + H

merged_df["new_genre"] = np.where(
    (merged_df["genre"] == A) | (merged_df["genre"] == B),
    C,
    merged_df["genre"]
)


merged_df["new_genre"] = np.where(
    (merged_df["genre"] == D) | (merged_df["genre"] == E),
    F,
    merged_df["new_genre"]
)

merged_df["new_genre"] = np.where(
    (merged_df["genre"] == G) | (merged_df["genre"] == H),
    I,
    merged_df["new_genre"]
)

train_df, test_df = devide_merged_df(merged_df)

# kNNのparametersを設定
N_CLASSES = len(train_df["new_genre"].unique())
n_neighbors = 6 # 近傍とみなすポイントの数
target = train_df["new_genre"]

model = KNeighborsClassifier(n_neighbors=n_neighbors + 1, weights="distance")

X = train_df[features].fillna(0.0).values * feature_weights

model.fit(X, target)

# 距離とインデックス配列を取得
distances, indexes = model.kneighbors(X)

# 配列から自分の距離(0)を取り除く
distances = distances[:, 1:]
indexes = indexes[:, 1:]
# 近傍6ポイントのgenre配列の作成
labels = target.values[indexes]

# predsを初期化
knn_pred_1st = []

# forループを使ってリスト内包表記の処理を展開
for labels_, distances_ in zip(labels, 1.0 / distances):

    bincount_result = np.bincount(labels_, weights=distances_, minlength=N_CLASSES)
    knn_pred_1st.append(bincount_result.argmax())

# predsをNumPy配列に変換
knn_pred_1st = np.array(knn_pred_1st)

# F1スコアの計算
score = f1_score(target, knn_pred_1st, average="macro")
print(f"f1_score = {score}")
print(classification_report(target, knn_pred_1st))

# 関数の呼び出し
visualize_confusion_matrix(target, knn_pred_1st)

# テストデータを予測
model = KNeighborsClassifier(n_neighbors)
# データセットの作成
X_train = train_df[features].fillna(0.0).values * feature_weights
X_test = test_df[features].fillna(0.0).values * feature_weights
# モデル学習
model.fit(X_train, target)
# 予測
test_df["genre"] = model.predict(X_test)

# feature weights

# one-hotエンコーディングした地域と、スケーリングした特徴量を変数化
features = [col for col in train_df.columns if col.startswith("region_") or col.startswith("standard")]

# 空の辞書を作成
dict_feature_weights = {}

# regionの重みを設定
for col in train_df.columns:
    if col.startswith("region_"):
        dict_feature_weights[col] = 100.0

# 楽曲成分特徴量の重み設定
for col in train_df.columns:
    if col.startswith("standard"):
        dict_feature_weights[col] = 1.0

# 別途重みを設定
dict_feature_weights["standardscaled_popularity"] = 3.75
# dict_feature_weights["standardscaled_log_tempo"] = 0.001
dict_feature_weights["standardscaled_log_tempo"] = 0.001
dict_feature_weights["standardscaled_num_nans"] = 100.0

# 重みを配列に
feature_weights = np.array([dict_feature_weights[col] for col in features])

train_df["first_knn_pred"] = knn_pred_1st
# # 1st_knn_predが18のみのデータフレームを作成
train_18 = train_df.query("first_knn_pred == 18")
test_18 = test_df.query("genre == 18")

# kNNのparametersを設定
N_CLASSES = len(train_18["genre"].unique())
n_neighbors = 2
target = train_18["genre"]
# モデルインスタンスの作成
model = KNeighborsClassifier(n_neighbors=n_neighbors + 1, weights="distance")
# 学習データセットの作成
x_train = train_18[features].fillna(0.0).values * feature_weights
# モデル学習
model.fit(x_train, target)
# 距離とインデックス配列を取得
distances, indexes = model.kneighbors(x_train)
# 配列から自分の距離(0)を取り除く
distances = distances[:, 1:]
indexes = indexes[:, 1:]
# 近傍6ポイントのgenre配列の作成
labels = target.values[indexes]

# predsを初期化
preds_18 = []

# forループを使ってリスト内包表記の処理を展開
for labels_, distances_ in zip(labels, 1.0 / distances):
    bincount_result = np.bincount(labels_, weights=distances_, minlength=N_CLASSES)
    # bincount_resultの中で最頻値（最大のインデックス）を取得してpredsに追加
    preds_18.append(bincount_result.argmax())

# predsをNumPy配列に変換
preds_18 = np.array(preds_18)

# F1スコアの計算
score = f1_score(target, preds_18, average="macro")
print(f"f1_score = {score}")
print(classification_report(target, preds_18))

model = KNeighborsClassifier(n_neighbors)
# データセットの作成
X_train = train_18[features].fillna(0.0).values * feature_weights
X_test = test_18[features].fillna(0.0).values * feature_weights
# モデル学習
model.fit(X_train, target)
# 予測
test_18["original_genre"] = model.predict(X_test)

test_18["original_genre"].value_counts().sort_index()
```


</details>

**・Pycaret, HGBC, OVR, lightgbm, xgb, SVC, LR, RFなどのモデルチューニング(optunaや重み調整等)**

</details>


# まとめ
フォーラムで紹介されていたkNNモデルを基に、さらに精度を高めるためにアンサンブルを試みました。同一モデルを使用しつつrandom_stateのみを変更する方法でアンサンブルを構築したところ、試行した中では最も高いOOFスコアを記録しました。リーダーボードのスコアとは乖離があったものの、自分のCV結果を信じた選択が最良だったと感じています。

結果として上位入賞には届きませんでしたが、不均衡データのマルチクラス分類に初めて取り組み、スコア改善にはつながらなかったものの、新たな知識（疑似ラベルやSMOTEなど）やアプローチ（OvRモデルなど）に触れることができ、とても有意義で楽しい経験となりました。
