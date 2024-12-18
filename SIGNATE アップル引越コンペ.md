# はじめに
機械学習の勉強を始めて約半年、書籍やスクールで学んだ内容がどの程度身についているのか、現時点での実力を確認する為、SIGNATEの[アップル引越し需要予測](https://signate.jp/competitions/269)へ挑戦。
今後データサイエンティストへ転職するにあたり、時系列データの回帰問題の知識を身に付けたいと思い、このコンペへ挑戦。

結果は最終評価 : 〇〇〇で、2024年12月19日時点で、1153人中 **〇〇位**でした。


# 実行環境

| カテゴリー | 名称 | バージョン　|
|:-:|:-:|:-:|
|パソコン|Surface Pro 6|10.0.19045 ビルド 19045　|
|開発環境|Visual Studio Code|1.94.2 (user setup)　|
|言語|Python|3.9.1|
|ライブラリ|lightgbm|4.3.0|
|ライブラリ|prophet|1.1.6|


# 目次
[1. データの確認](#データの確認) 
[2. データセットの作成](#データセットの作成) 
[3. LightGBMでのモデル構築](#LightGBMでのモデル構築) 
[4. Prophetでのモデル構築](#Prophetでのモデル構築)
[5. 予測値の描画](#予測値の描画)
[6. アンサンブル](#アンサンブル)
[7. まとめ](#まとめ)





# データの確認
**ライブラリのインポート**

```python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

import sweetviz as sv

from prophet import Prophet

from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error

import lightgbm as lgb

```

**データ読み込み&構造確認**
```python
# パスの指定
path_train = os.path.join(r"Data\train.csv")
path_test = os.path.join(r"Data\test.csv")

# データの読み込み
train_df = pd.read_csv(path_train)
test_df = pd.read_csv(path_test)

# データフレームに名前を付ける
train_df.name = "Train_df"
test_df.name = "Test_df"

# データ構造の確認
print(train_df.name, train_df.shape)
print(test_df.name, test_df.shape)
```
```python
# 実行結果
Train_df (2101, 6)
Test_df (365, 5)
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
実行結果
|   | col       | Null_tr | Dtype_tr | Null_te | Dtype_te |
|---|-----------|---------|----------|---------|----------|
| 0 | datetime  | 0       | object   | 0       | object   |
| 1 | y         | 0       | int64    | NaN     | NaN      |
| 2 | client    | 0       | int64    | 0       | int64    |
| 3 | close     | 0       | int64    | 0       | int64    |
| 4 | price_am  | 0       | int64    | 0       | int64    |
| 5 | price_pm  | 0       | int64    | 0       | int64    |

**相関係数の確認**
```python
# 2列目以降の列で相関行列を計算
cor = train_df.iloc[:, 1:].corr()
# 図の表示サイズを指定
plt.figure(figsize=(8,8))
# 描画
sns.heatmap(cor, cmap= sns.color_palette('coolwarm', 10), annot=True,fmt='.2f', vmin = -1, vmax = 1)
```
![output.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3794471/c45f1ce3-4902-a880-73aa-47ab454dd3a5.png)

**データの概要を確認**
```python
# sweetvizの実行
report = sv.compare(train_df, test_df, target_feat="y")
# 結果をhtml形式で保存
report.show_html("report.html")
```

# データセットの作成

```python
def make_dataset():

    # パスの指定
    path_train = os.path.join(r"Data\train.csv")
    path_test = os.path.join(r"Data\test.csv")

    # データの読み込み
    train_df = pd.read_csv(path_train)
    test_df = pd.read_csv(path_test)

    # price_amが-1ならprice_pmで補完、price_pmが-1ならprice_amで補完
    train_df["price_am"] = np.where(train_df["price_am"] == -1, train_df["price_pm"], train_df["price_am"])
    train_df["price_pm"] = np.where(train_df["price_pm"] == -1, train_df["price_am"], train_df["price_pm"])

    # テストデータも同様の処理を行う

    # price_amが-1ならprice_pmで補完、price_pmが-1ならprice_amで補完
    test_df["price_am"] = np.where(test_df["price_am"] == -1, test_df["price_pm"], test_df["price_am"])
    test_df["price_pm"] = np.where(test_df["price_pm"] == -1, test_df["price_am"], test_df["price_pm"])

    # テストデータの場合は欠損値が前半に集中しているとかは無いので、-1を0に置き換える
    test_df["price_am"] = np.where(test_df["price_am"] == -1, 0, test_df["price_am"])
    test_df["price_pm"] = np.where(test_df["price_pm"] == -1,  0, test_df["price_pm"])

    # 日付から簡単な特徴量作成
    train_df["date"] = pd.to_datetime(train_df["datetime"])
    test_df["date"] = pd.to_datetime(test_df["datetime"])

    train_df["year"] = train_df["date"].dt.year
    test_df["year"] = test_df["date"].dt.year

    train_df["month"] = train_df["date"].dt.month
    test_df["month"] = test_df["date"].dt.month

    train_df["day"] = train_df["date"].dt.day
    test_df["day"] = test_df["date"].dt.day

    train_df["weekday"] = train_df["date"].dt.weekday
    test_df["weekday"] = test_df["date"].dt.weekday

    train_df = train_df.drop(["date"], axis=1)
    test_df = test_df.drop(["date"], axis=1)

    return train_df, test_df
```


# LightGBMでのモデル構築

```python
train_df, test_df = make_dataset()

# データセットの作成
x_train = train_df.drop(["datetime", "y"], axis=1)
y_train = train_df["y"]
id_train = train_df[["datetime"]]

x_test = test_df.drop(["datetime"], axis=1)
id_test = test_df[["datetime"]]

# データの分割
x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# LightGBM分類器のインスタンスを作成
lgb_reg = lgb.LGBMRegressor(random_state=42, verbose=-1)

# クロスバリデーションの分割方法を設定
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# クロスバリデーションで分類器を学習
y_pred = cross_val_predict(lgb_reg, x_tr, y_tr, cv=cv)

# 学習データの性能評価(MAE)
print("MAE: ", mean_absolute_error(y_tr, y_pred))

# クロスバリデーションとは別に再学習
lgb_reg.fit(x_tr, y_tr)

# テストデータに対する予測
y_va_pred = lgb_reg.predict(x_va)

# 予測の性能評価
print("VA MAE: ", mean_absolute_error(y_va, y_va_pred))

# 特徴量の重要度を変数に格納
feature_importances = lgb_reg.feature_importances_

# カラム名と共に表示
imp = pd.DataFrame({
    "feature": x_train.columns,
    "imp": feature_importances
}).sort_values(ascending=False, by="imp")

display(imp)

# テストデータに対する予測
lgb_test_pred = lgb_reg.predict(x_test)
lgb_result_df = pd.DataFrame({
    "datetime": id_test["datetime"],
    "close" : x_test["close"],
    "y": lgb_test_pred
})

# close=1の予測値を0に置き換える
lgb_result_df["y"] = np.where(lgb_result_df["close"] == 1, 0, lgb_result_df["y"])

# submit用にcsv出力
lgb_result_df = lgb_result_df[["datetime", "y"]]
lgb_result_df.to_csv("lgb_pred.csv", index=False, header=False)
```
MAE:  5.21945486994995
VA MAE:  5.06855370123543
| feature   |   imp |
|:----------|------:|
| day       |   903 |
| month     |   838 |
| year      |   420 |
| weekday   |   370 |
| price_am  |   221 |
| price_pm  |   153 |
| client    |    49 |
| close     |    46 |

# Prophetでのモデル構築
```python
train_df, test_df = make_dataset()

# データセット
prop_train_df = train_df.copy().rename(columns={"datetime" : "ds"})

# モデル構築
model_normal_multipli = Prophet(
    seasonality_mode="multiplicative",
    changepoint_prior_scale = 0.5,
    yearly_seasonality=True,
    weekly_seasonality=True,
    growth="linear"
)

# 学習
normal_model = model_normal_multipli.fit(prop_train_df)

# 予測期間の設定
future = model_normal_multipli.make_future_dataframe(periods=365, freq="D")

# 予測
forecast_normal_multipli = model_normal_multipli.predict(future)

# 予測結果をグラフで描画
plt_normal_multipli = model_normal_multipli.plot(forecast_normal_multipli)

# submit用にデータフレームを調整
prop_pred = forecast_normal_multipli.tail(365).copy().reset_index(drop=True)
prop_result_df = pd.DataFrame({
    "datetime" : test_df["datetime"],
    "close" : test_df["close"],
    "y" : prop_pred["yhat"]
})
prop_result_df["y"] = np.where(prop_result_df["close"] == 1, 0, prop_result_df["y"])
prop_result_df["datetime"] = pd.to_datetime(prop_result_df["datetime"])
prop_result_df[["datetime", "y"]].to_csv("prop_pred.csv", index=False, header=False)
```
![output12.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3794471/d2b6856e-7aff-635e-f224-0932d51927ef.png)

```python
# 各成分を可視化
normal_model.plot_components(forecast_normal_multipli)
```
![output13.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3794471/063509de-4ebe-3d9d-6d62-a0a65b4b5300.png)



# 予測値の描画

```python
lgb_result_df["datetime"] = pd.to_datetime(lgb_result_df["datetime"])
train_df["datetime"] = pd.to_datetime(train_df["datetime"])
sns.lineplot(lgb_result_df, x="datetime", y="y", label="LightGBM")
sns.lineplot(prop_result_df, x="datetime", y="y", label="Prophet")
```
![output14.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3794471/f82e5609-bbd9-8363-f9cd-db70d2b20614.png)


# アンサンブル

```python
# アンサンブル用データフレームの作成
ensemble_lgb_prop_df = pd.DataFrame({
    "datetime" : lgb_result_df["datetime"],
    "month" : test_df["month"],
    "lgb_pred" : lgb_result_df["y"],
    "prop_pred" : prop_result_df["y"]
})

# 3月のデータのみprophetの予測値に置き換え
ensemble_lgb_prop_df["ensemble_pred"] = np.where(ensemble_lgb_prop_df["month"] == 3, ensemble_lgb_prop_df["prop_pred"], ensemble_lgb_prop_df["lgb_pred"])

# csv出力
ensemble_lgb_prop_df[["datetime", "ensemble_pred"]].to_csv("ensemble_lgb_prop.csv", index=False, header=False)
```
# まとめ
LightGBMによる予測ではトレンドを十分に反映できていないと感じたため、3月分のみProphetの予測値に置き換えました。このアプローチが効果を発揮したように思います。ただ、3月の予測値はもう少し高くなると予想しており、予測精度をさらに向上させるために、パラメータのチューニングやその他のモデル構築も試みましたが、意図するモデルを構築する事が出来なかったため、今後の課題だと考える。

今回は過去問を使っての実践でしたが、実際のコンペ形式での挑戦は初めてでした。モデル構築の手順やエラーへの対処法、未知のモデルや手法について多くの学びがあり、非常に有意義な経験となりました。
