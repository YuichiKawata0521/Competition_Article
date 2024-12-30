# はじめに
初めてリアルタイムで開催されているコンペ、SIGNATEの[SMBC Group GREEN×DATA Challenge 2024](https://signate.jp/competitions/1491)へ挑戦しました。
結果は862人中**144位**で、上位約17%に入りました。
![SMBCリーダーボード.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3794471/074e43f2-a58c-c86a-fcba-c1d7e96aedc2.png)


# 実行環境

| カテゴリー | 名称 | バージョン　|
|:-:|:-:|:-:|
|パソコン|Surface Pro 6|10.0.19045 ビルド 19045　|
|開発環境|Visual Studio Code|1.94.2 (user setup)　|
|言語|Python|3.9.1|
|ライブラリ|scikit-learn|1.4.2|
|ライブラリ|xgboost|2.1.1|

# 目次
[1. データの確認](#データの確認) 

[2. 関数の定義](#関数の定義) 

[3. 各モデルのパラメータ調整](#各モデルのパラメータ調整)

[4. アンサンブル](#アンサンブル)

[5. まとめ](#まとめ)

# データの確認
```python
### オリジナルデータ読み込み関数
def load_original_data():
    # パスの指定
    path_train = os.path.join(r"Data\train.csv")
    path_test = os.path.join(r"Data\test.csv")

    # データの読み込み
    train_df = pd.read_csv(path_train, index_col=0)
    test_df = pd.read_csv(path_test, index_col=0)

    # データ構造の確認
    print("★オリジナルデータ読み込み時")
    print("train_df.shape", train_df.shape)
    print("test_df.shape", test_df.shape)
    return train_df, test_df
```
```python
### 欠損値とデータ型確認関数
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
```
**読み込みと確認**
```python
# オリジナルデータの読み込み
train_df, test_df = load_original_data()
# 先頭5行表示
display(train_df.head())
# 欠損値＆データ型の確認
check_null(train_df, test_df)
```
**出力結果**
```
★オリジナルデータ読み込み時
train_df.shape (4655, 21)
test_df.shape (2508, 20)
```
(先頭5行)
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FacilityName</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>LocationAddress</th>
      <th>City</th>
      <th>State</th>
      <th>ZIP</th>
      <th>County</th>
      <th>FIPScode</th>
      <th>PrimaryNAICS</th>
      <th>SecondPrimaryNAICS</th>
      <th>IndustryType</th>
      <th>TRI_Air_Emissions_10_in_lbs</th>
      <th>TRI_Air_Emissions_11_in_lbs</th>
      <th>TRI_Air_Emissions_12_in_lbs</th>
      <th>TRI_Air_Emissions_13_in_lbs</th>
      <th>GHG_Direct_Emissions_10_in_metric_tons</th>
      <th>GHG_Direct_Emissions_11_in_metric_tons</th>
      <th>GHG_Direct_Emissions_12_in_metric_tons</th>
      <th>GHG_Direct_Emissions_13_in_metric_tons</th>
      <th>GHG_Direct_Emissions_14_in_metric_tons</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>VISCOFAN USA INC</td>
      <td>40.141389</td>
      <td>-87.581111</td>
      <td>915 N MICHIGAN AVE</td>
      <td>DANVILLE</td>
      <td>IL</td>
      <td>61832</td>
      <td>VERMILION</td>
      <td>17183.0</td>
      <td>326121</td>
      <td>NaN</td>
      <td>Other</td>
      <td>31566.709644</td>
      <td>26644.986107</td>
      <td>23410.379903</td>
      <td>31809.857564</td>
      <td>64816.958901</td>
      <td>36588.744606</td>
      <td>37907.936721</td>
      <td>45598.125851</td>
      <td>52973.139946</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CORNWELL STATION - DOMINION TRANSMISSION, INC</td>
      <td>38.475305</td>
      <td>-81.278957</td>
      <td>200 RIVER HAVEN ROAD</td>
      <td>CLENDENIN</td>
      <td>WV</td>
      <td>25045-9304</td>
      <td>KANAWHA</td>
      <td>54039.0</td>
      <td>486210</td>
      <td>NaN</td>
      <td>Petroleum and Natural Gas Systems</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>55507.543666</td>
      <td>72387.334115</td>
      <td>58225.196089</td>
      <td>76376.547318</td>
      <td>55910.066617</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WEST UNION COMPRESSOR STATION</td>
      <td>39.299820</td>
      <td>-80.857170</td>
      <td>3041 LONG RUN RD.</td>
      <td>GREENWOOD</td>
      <td>WV</td>
      <td>26415</td>
      <td>RITCHIE</td>
      <td>54085.0</td>
      <td>211112</td>
      <td>NaN</td>
      <td>Petroleum and Natural Gas Systems</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>55679.543214</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DOUBLE "C" LIMITED</td>
      <td>35.490363</td>
      <td>-119.042957</td>
      <td>10245 OILFIELD ROAD</td>
      <td>BAKERSFIELD</td>
      <td>CA</td>
      <td>93308</td>
      <td>KERN</td>
      <td>6029.0</td>
      <td>221112</td>
      <td>NaN</td>
      <td>Power Plants</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>54811.222708</td>
      <td>69339.923002</td>
      <td>63647.340038</td>
      <td>53799.011225</td>
      <td>61411.902782</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LAKESHORE PLANT</td>
      <td>41.532802</td>
      <td>-81.636448</td>
      <td>6800 S MARGINAL RD</td>
      <td>CLEVELAND</td>
      <td>OH</td>
      <td>441031047</td>
      <td>CUYAHOGA</td>
      <td>39035.0</td>
      <td>221112</td>
      <td>NaN</td>
      <td>Power Plants</td>
      <td>29553.796627</td>
      <td>28337.832145</td>
      <td>30840.825454</td>
      <td>25153.901905</td>
      <td>81812.306362</td>
      <td>53823.561587</td>
      <td>77391.157768</td>
      <td>17662.966241</td>
      <td>43100.469774</td>
    </tr>
    <!-- 横幅を確保するためのダミー行 -->
<tfoot>
<tr><th colspan=100>$\hspace{250em}$</th></tr>
</tfoot>
  </tbody>
</table>

(欠損値とデータ型)
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col</th>
      <th>Null_tr</th>
      <th>Dtype_tr</th>
      <th>Null_te</th>
      <th>Dtype_te</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FacilityName</td>
      <td>0</td>
      <td>object</td>
      <td>0</td>
      <td>object</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Latitude</td>
      <td>102</td>
      <td>float64</td>
      <td>56</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Longitude</td>
      <td>102</td>
      <td>float64</td>
      <td>56</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LocationAddress</td>
      <td>179</td>
      <td>object</td>
      <td>113</td>
      <td>object</td>
    </tr>
    <tr>
      <th>4</th>
      <td>City</td>
      <td>0</td>
      <td>object</td>
      <td>0</td>
      <td>object</td>
    </tr>
    <tr>
      <th>5</th>
      <td>State</td>
      <td>0</td>
      <td>object</td>
      <td>0</td>
      <td>object</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ZIP</td>
      <td>0</td>
      <td>object</td>
      <td>0</td>
      <td>object</td>
    </tr>
    <tr>
      <th>7</th>
      <td>County</td>
      <td>70</td>
      <td>object</td>
      <td>45</td>
      <td>object</td>
    </tr>
    <tr>
      <th>8</th>
      <td>FIPScode</td>
      <td>73</td>
      <td>float64</td>
      <td>45</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PrimaryNAICS</td>
      <td>0</td>
      <td>int64</td>
      <td>0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>10</th>
      <td>SecondPrimaryNAICS</td>
      <td>4276</td>
      <td>float64</td>
      <td>2324</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>11</th>
      <td>IndustryType</td>
      <td>1</td>
      <td>object</td>
      <td>0</td>
      <td>object</td>
    </tr>
    <tr>
      <th>12</th>
      <td>TRI_Air_Emissions_10_in_lbs</td>
      <td>3020</td>
      <td>float64</td>
      <td>1634</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>13</th>
      <td>TRI_Air_Emissions_11_in_lbs</td>
      <td>3020</td>
      <td>float64</td>
      <td>1634</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>14</th>
      <td>TRI_Air_Emissions_12_in_lbs</td>
      <td>3020</td>
      <td>float64</td>
      <td>1634</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>15</th>
      <td>TRI_Air_Emissions_13_in_lbs</td>
      <td>3020</td>
      <td>float64</td>
      <td>1634</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>16</th>
      <td>GHG_Direct_Emissions_10_in_metric_tons</td>
      <td>702</td>
      <td>float64</td>
      <td>378</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GHG_Direct_Emissions_11_in_metric_tons</td>
      <td>371</td>
      <td>float64</td>
      <td>211</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>18</th>
      <td>GHG_Direct_Emissions_12_in_metric_tons</td>
      <td>260</td>
      <td>float64</td>
      <td>137</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GHG_Direct_Emissions_13_in_metric_tons</td>
      <td>148</td>
      <td>float64</td>
      <td>73</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>20</th>
      <td>GHG_Direct_Emissions_14_in_metric_tons</td>
      <td>0</td>
      <td>float64</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <!-- 横幅を確保するためのダミー行 -->
<tfoot>
<tr><th colspan=6>$\hspace{37em}$</th></tr>
</tfoot>

  </tbody>
</table>

**sweetvizの実行**
```python
# sweetvizの実行
report = sv.compare(train_df, test_df, target_feat="GHG_Direct_Emissions_14_in_metric_tons")
# 結果をhtml形式で保存
report.show_html("report.html")
```
**要約統計量の確認**
```python
# 数値の表示フォーマットを設定
pd.set_option('display.float_format', '{:.3f}'.format)

# 要約統計量の確認
print("★★ train_df ★★")
display(train_df.iloc[:, 12:].describe())
print("★★ test_df ★★")
test_df.iloc[:, 12:].describe()
```
★★ train_df ★★
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TRI_Air_Emissions_10_in_lbs</th>
      <th>TRI_Air_Emissions_11_in_lbs</th>
      <th>TRI_Air_Emissions_12_in_lbs</th>
      <th>TRI_Air_Emissions_13_in_lbs</th>
      <th>GHG_Direct_Emissions_10_in_metric_tons</th>
      <th>GHG_Direct_Emissions_11_in_metric_tons</th>
      <th>GHG_Direct_Emissions_12_in_metric_tons</th>
      <th>GHG_Direct_Emissions_13_in_metric_tons</th>
      <th>GHG_Direct_Emissions_14_in_metric_tons</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1635.000</td>
      <td>1635.000</td>
      <td>1635.000</td>
      <td>1635.000</td>
      <td>3953.000</td>
      <td>4284.000</td>
      <td>4395.000</td>
      <td>4507.000</td>
      <td>4655.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>61790.639</td>
      <td>43853.462</td>
      <td>53770.293</td>
      <td>56007.086</td>
      <td>248515.825</td>
      <td>161206.839</td>
      <td>315990.492</td>
      <td>183400.568</td>
      <td>252513.310</td>
    </tr>
    <tr>
      <th>std</th>
      <td>134498.337</td>
      <td>55988.952</td>
      <td>93977.128</td>
      <td>109863.243</td>
      <td>522511.025</td>
      <td>264183.085</td>
      <td>739584.332</td>
      <td>402623.710</td>
      <td>485466.928</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2279.515</td>
      <td>34.451</td>
      <td>2076.649</td>
      <td>4656.523</td>
      <td>108.941</td>
      <td>0.817</td>
      <td>200.997</td>
      <td>26.893</td>
      <td>559.807</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25252.148</td>
      <td>25319.053</td>
      <td>22764.672</td>
      <td>22905.881</td>
      <td>51239.753</td>
      <td>37339.350</td>
      <td>48224.873</td>
      <td>35477.808</td>
      <td>41748.692</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>32041.870</td>
      <td>31765.720</td>
      <td>29667.092</td>
      <td>29305.094</td>
      <td>74403.472</td>
      <td>61197.488</td>
      <td>72426.838</td>
      <td>57446.583</td>
      <td>67897.929</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>40191.233</td>
      <td>38329.096</td>
      <td>36984.432</td>
      <td>36755.018</td>
      <td>167640.863</td>
      <td>141860.781</td>
      <td>227966.788</td>
      <td>119554.357</td>
      <td>210916.781</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1251231.476</td>
      <td>478366.459</td>
      <td>743548.788</td>
      <td>989230.803</td>
      <td>3900222.229</td>
      <td>2698567.311</td>
      <td>6837259.915</td>
      <td>4330235.707</td>
      <td>4614102.600</td>
    </tr>
    <tfoot>
<tr><th colspan=100>$\hspace{145em}$</th></tr>
</tfoot>
  </tbody>
</table>


★★ test_df ★★
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TRI_Air_Emissions_10_in_lbs</th>
      <th>TRI_Air_Emissions_11_in_lbs</th>
      <th>TRI_Air_Emissions_12_in_lbs</th>
      <th>TRI_Air_Emissions_13_in_lbs</th>
      <th>GHG_Direct_Emissions_10_in_metric_tons</th>
      <th>GHG_Direct_Emissions_11_in_metric_tons</th>
      <th>GHG_Direct_Emissions_12_in_metric_tons</th>
      <th>GHG_Direct_Emissions_13_in_metric_tons</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>874.000</td>
      <td>874.000</td>
      <td>874.000</td>
      <td>874.000</td>
      <td>2130.000</td>
      <td>2297.000</td>
      <td>2371.000</td>
      <td>2435.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>64875.128</td>
      <td>40970.200</td>
      <td>49969.491</td>
      <td>50999.788</td>
      <td>256710.617</td>
      <td>160143.192</td>
      <td>312410.186</td>
      <td>189274.471</td>
    </tr>
    <tr>
      <th>std</th>
      <td>153260.893</td>
      <td>50742.809</td>
      <td>96607.925</td>
      <td>109221.257</td>
      <td>523900.691</td>
      <td>275363.267</td>
      <td>744804.547</td>
      <td>428452.150</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4405.829</td>
      <td>5343.549</td>
      <td>3144.557</td>
      <td>1221.438</td>
      <td>134.226</td>
      <td>52.038</td>
      <td>3003.952</td>
      <td>222.638</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>24657.268</td>
      <td>25406.684</td>
      <td>23070.545</td>
      <td>22060.824</td>
      <td>50253.701</td>
      <td>35439.342</td>
      <td>45703.760</td>
      <td>35078.160</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>31372.262</td>
      <td>30861.834</td>
      <td>29153.614</td>
      <td>28119.963</td>
      <td>73134.349</td>
      <td>58628.967</td>
      <td>69985.577</td>
      <td>56596.058</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>38983.842</td>
      <td>37976.076</td>
      <td>35656.638</td>
      <td>34989.526</td>
      <td>185099.584</td>
      <td>113888.966</td>
      <td>215016.871</td>
      <td>111387.358</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1374502.927</td>
      <td>481505.853</td>
      <td>898432.732</td>
      <td>1010901.828</td>
      <td>3927869.397</td>
      <td>2120410.856</td>
      <td>6112087.346</td>
      <td>4292471.520</td>
    </tr>
    <tfoot>
<tr><th colspan=100>$\hspace{128em}$</th></tr>
</tfoot>
  </tbody>
</table>

# 関数の定義
```python
# merged_df作成関数
def make_merged_df(train_df, test_df):
    train_df["train"] = 1
    test_df["train"] = 0
    test_df["GHG_Direct_Emissions_14_in_metric_tons"] = 0
    merged_df = pd.concat([train_df, test_df], axis=0)
    print("★merged_df作成時")
    print("merged_df.shape : ", merged_df.shape)
    return merged_df

# merged_df分割関数
def devide_merged_df(merged_df):
    # trainとtestデータに分割
    train_df = merged_df[merged_df["train"] == 1].copy()
    test_df = merged_df[merged_df["train"] == 0].copy()
    # 不要列を削除
    train_df.drop(["train"], axis=1, inplace=True)
    test_df.drop(["GHG_Direct_Emissions_14_in_metric_tons", "train"], axis=1, inplace=True)
    print("★merged_dfを分割時")
    print("train_df.shape : ", train_df.shape)
    print("test_df.shape : ", test_df.shape)
    return train_df, test_df
```
```python
two_digit_map     = {11: 'Agriculture, Forestry, Fishing and Hunting',
                    21: 'Mining, Quarrying, and Oil and Gas Extraction',
                    22: 'Utilities',
                    23: 'Construction',
                    31: 'Manufacturing',
                    32: 'Manufacturing',
                    33: 'Manufacturing',
                    42: 'Wholesale Trade',
                    44: 'Retail Trade',
                    45: 'Retail Trade',
                    48: 'Transportation and Warehousing',
                    49: 'Transportation and Warehousing',
                    51: 'Information',
                    52: 'Finance and Insurance',
                    53: 'Real Estate and Rental and Leasing',
                    54: 'Professional, Scientific, and Technical Services',
                    55: 'Management of Companies and Enterprises',
                    56: 'Administrative and Support and Waste Management and Remediation Services',
                    61: 'Educational Services',
                    62: 'Health Care and Social Assistance',
                    71: 'Arts, Entertainment, and Recreation',
                    72: 'Accommodation and Food Services',
                    81: 'Other Services (except Public Administration)',
                    92: 'Public Administration'}
# PrimaryNAICSの最初の2桁からEconomic_Sectorを抽出し、不要な中間カラムは削除
def make_sector(train_df, test_df):
    merged_df = make_merged_df(train_df, test_df)

    merged_df['Economic_Sector'] = merged_df['PrimaryNAICS'].apply(lambda z: two_digit_map[int(str(z)[:2])])
    merged_df.drop(columns=["PrimaryNAICS", "SecondPrimaryNAICS", "IndustryType"], inplace=True)
    return merged_df
```
```python
def make_features(merged_df):
    # 特徴量作成
    for i in [10, 11, 12]:
        ### GHG特徴特徴量
        # 前年差異
        merged_df[f"GHG{i + 1}_minus_GHG{i}"] = merged_df[f"GHG_Direct_Emissions_{i + 1}_in_metric_tons"] - merged_df[f"GHG_Direct_Emissions_{i}_in_metric_tons"]
        # 前年増減率
        mask = (merged_df[f"GHG_Direct_Emissions_{i}_in_metric_tons"].notna()) &(merged_df[f"GHG_Direct_Emissions_{i}_in_metric_tons"] != 0)
        merged_df[f"GHG{i + 1}_Growth_rate"] = np.where(
            mask,
            (merged_df[f"GHG_Direct_Emissions_{i + 1}_in_metric_tons"] - merged_df[f"GHG_Direct_Emissions_{i}_in_metric_tons"]) / merged_df[f"GHG_Direct_Emissions_{i}_in_metric_tons"],
            np.nan)
        
    # 必要な列のみを抽出
    new_merged_df = merged_df.iloc[:, 9:].copy()

    le = LabelEncoder()
    new_merged_df["Economic_Sector"] = le.fit_transform(new_merged_df["Economic_Sector"])

    return new_merged_df
```




# 各モデルのパラメータ調整
## HGBR
```python
# オリジナルデータの読み込み
train_df, test_df = load_original_data()
# Economic_Sectorを作成
merged_df = make_sector(train_df, test_df)
# 特徴量作成
new_merged_df = make_features(merged_df)
# データを分割
train_df, test_df = devide_merged_df(new_merged_df)

# データの準備
hgbr_train_df = train_df.copy().reset_index(drop=False)
hgbr_test_df = test_df.copy().reset_index(drop=False)
# データセットの作成
hgbr_x_train = hgbr_train_df.drop(columns=["GHG_Direct_Emissions_14_in_metric_tons", "index"])
hgbr_y_train = hgbr_train_df["GHG_Direct_Emissions_14_in_metric_tons"]
hgbr_id_train = hgbr_train_df[["index"]]
hgbr_x_test = hgbr_test_df.drop(columns="index")
hgbr_id_test = hgbr_test_df[["index"]]

# データ分割
x_tr, x_va, y_tr, y_va = train_test_split(hgbr_x_train, hgbr_y_train, test_size=0.2, shuffle=True, random_state=42)

# Optunaの目的関数
def objective(trial):
    # ハイパーパラメータの探索範囲
    params = {
        'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'l2_regularization': trial.suggest_loguniform('l2_regularization', 1e-6, 1e-1),
        'max_bins': trial.suggest_int('max_bins', 2, 255),
        "random_state" : 42
    }
    
    # モデルの定義
    model = HistGradientBoostingRegressor(**params)
    
    # モデルの学習
    model.fit(x_tr, y_tr)
    
    # 予測
    y_pred = model.predict(x_va)
    y_pred = np.maximum(y_pred, 0)

    # RMSLEで評価
    score = root_mean_squared_log_error(y_va, y_pred)
    return score

# Optunaでのハイパーパラメータチューニング
study = optuna.create_study(direction='minimize')  # 最小化問題
study.optimize(objective, n_trials=100)  # 試行回数50回

# 最適なパラメータの出力
print(f"Best hyperparameters: {study.best_params}")
```
```
Best hyperparameters: {'max_iter': 500, 'learning_rate': 0.0124, 'max_depth': 10, 'min_samples_leaf': 7, 'l2_regularization': 0.00061, 'max_bins': 188, 'random_state': 42}
```
## XGB_1 & 2
```python
# データの準備
xgb_train_df = train_df.copy().reset_index(drop=False)
xgb_test_df = test_df.copy().reset_index(drop=False)
# データセットの作成
xgb_x_train = xgb_train_df.drop(columns=["GHG_Direct_Emissions_14_in_metric_tons", "index"])
xgb_y_train = xgb_train_df["GHG_Direct_Emissions_14_in_metric_tons"]
xgb_id_train = xgb_train_df[["index"]]
xgb_x_test = xgb_test_df.drop(columns="index")
xgb_id_test = xgb_test_df[["index"]]

# Optunaの目的関数
def objective(trial):
    # 試行するハイパーパラメータを設定
    param = {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-3, 10.0),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "seed": 123, # XGB_1で使用
        "seed": 42 # XGB_2で使用
    }
    
    # モデルの作成と学習
    model = xgb.XGBRegressor(**param)
    model.fit(x_tr, y_tr)
    
    # 検証データで予測
    y_va_pred = model.predict(x_va)
    # 負の数を0に置き換え
    y_va_pred = np.maximum(y_va_pred, 0)
    
    # RMSLEを評価値として返す
    return root_mean_squared_log_error(y_va, y_va_pred)

# Optunaによるハイパーパラメータ探索
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# 最適なハイパーパラメータを出力
print("Best params: ", study.best_trial.params)
```
```
XGB_1
Best params: 'objective': "reg:squarederror", 'n_estimators': 600, 'learning_rate': 0.012, 'max_depth': 9, 'min_child_weight': 0.3366, 'subsample': 0.655, 'colsample_bytree': 0.741, 'reg_alpha': 0.2599, 'reg_lambda': 3.6082, 'seed': 123

XGB_2
Best params: 'objective': "reg:squarederror", 'n_estimators': 600, 'learning_rate': 0.0153, 'max_depth': 6, 'min_child_weight': 1.435, 'subsample': 0.774, 'colsample_bytree': 0.759, 'reg_alpha': 0.541, 'reg_lambda': 0.0383, 'seed': 42
```



# アンサンブル
(アンサンブル関数の定義)
```python
def train_and_predict(model, x_tr, y_tr, x_va, y_va, test):
    model.fit(x_tr, np.log1p(y_tr)) # 学習
    valid_preds = np.expm1(model.predict(x_va))  # 検証データを予測
    rmsle = root_mean_squared_log_error(y_va, valid_preds) # RMSLEを定義
    test_preds = model.predict(test) # テストデータの予測
    return valid_preds, rmsle, test_preds

def get_models_trained(train, test, target, numfolds):
    print("★分析開始")
    kf = KFold(n_splits=numfolds, shuffle=True, random_state=42) # クロスバリデーションの設定
    oof_predictions = np.zeros(len(train)) # OOF格納配列
    test_predictions = np.zeros(len(test)) # 予測値格納配列

    # アンサンブルするモデルの定義
    models = [
        ('model1', HistGradientBoostingRegressor(max_iter=500, learning_rate=0.0124, max_depth=10, min_samples_leaf=7, l2_regularization=0.00061, max_bins=188, random_state=42)),
        ('model2', xgb.XGBRegressor(objective="reg:squarederror", n_estimators=600, learning_rate=0.012, max_depth=9, min_child_weight=0.3366, subsample=0.655, colsample_bytree=0.741, reg_alpha=0.2599, reg_lambda=3.6082, seed=123)),
        ('model3', xgb.XGBRegressor(objective="reg:squarederror", n_estimators=600, learning_rate=0.0153, max_depth=6, min_child_weight=1.435, subsample=0.774, colsample_bytree=0.759, reg_alpha=0.541, reg_lambda=0.0383, seed=42))
    ]

    for fold, (train_index, valid_index) in enumerate(kf.split(train, target)):
        # 学習データと検証データに分割
        x_tr, x_va = train.iloc[train_index], train.iloc[valid_index]
        y_tr, y_va = target.iloc[train_index], target.iloc[valid_index]

        # 空の辞書とリストを作成
        valid_pred_dict = {}
        loss_dict = {}
        output_predictions = []

        # 各モデルで学習、検証＆予測
        for name, model in models:
            valid_preds, rmsle, test_preds = train_and_predict(model, x_tr, y_tr, x_va, y_va, test)
            valid_pred_dict[name] = valid_preds
            loss_dict[name] = rmsle
            output_predictions.append(test_preds)

        # 予測値の平均値を計算
        valid_preds_mean = np.mean(list(valid_pred_dict.values()), axis=0)
        # RMSLEの平均値を計算
        rmsle_mean = root_mean_squared_log_error(y_va, valid_preds_mean)

        # 最小の損失を持つモデルを選択
        min_loss_model = min(loss_dict, key=loss_dict.get)
        best_model = dict(models)[min_loss_model]
        valid_preds_best_model = np.expm1(best_model.predict(x_va))
        rmsle_best = root_mean_squared_log_error(y_va, valid_preds_best_model)

        # アンサンブル結果の方が良かった場合
        if rmsle_mean > rmsle_best: 
            print(f"Fold {fold+1}: Averaging all models. RMSLE mean: {rmsle_mean}, Best RMSLE: {rmsle_best}")
            test_predictions += np.mean(output_predictions, axis=0) / kf.n_splits
            oof_predictions[valid_index] = valid_preds_best_model
        # 単一モデル結果の方が良かった場合
        else:
            print(f"Fold {fold+1}: Using best model. RMSLE mean: {rmsle_mean}, Best RMSLE: {rmsle_best}")
            test_predictions += test_preds / kf.n_splits
            oof_predictions[valid_index] = valid_preds_best_model

        print('---------------\n')

    # OOF用のRMSLEを計算
    RMSLE = root_mean_squared_log_error(target, oof_predictions)
    print(f"OOF RMSLE = {RMSLE}")

    return oof_predictions, np.expm1(test_predictions)

```
(学習＆予測)
```python
# オリジナルデータの読み込み
train_df, test_df = load_original_data()
# Economic_Sectorを作成
merged_df = make_sector(train_df, test_df)
# 特徴量作成
new_merged_df = make_features(merged_df)
# データを分割
train_df, test_df = devide_merged_df(new_merged_df)
# データの準備
train = train_df.drop(columns=["GHG_Direct_Emissions_14_in_metric_tons"])
test = test_df
target = train_df["GHG_Direct_Emissions_14_in_metric_tons"]

# 学習と予測
oof_predictions,test_predictions = get_models_trained(train,test,target,3)

# 提出用dfの作成
ensemble_df = pd.DataFrame({
    "id" : test_df.index,
    "pred" : test_predictions
})

# csv出力
ensemble_df.to_csv(r"submit\ensemble_pred.csv", index=False, header=False)
```

```
オリジナルデータ読み込み時
train_df.shape (4655, 21)
test_df.shape (2508, 20)
merged_df作成時
merged_df.shape :  (7163, 22)
merged_dfを分割時
train_df.shape :  (4655, 16)
test_df.shape :  (2508, 15)
Fold 1: Using best model. RMSLE mean: 0.742953835124598, Best RMSLE: 0.743035564186474
---------------

Fold 2: Averaging all models. RMSLE mean: 0.8053936205746768, Best RMSLE: 0.8052706112536256
---------------

Fold 3: Using best model. RMSLE mean: 0.7778324631443038, Best RMSLE: 0.78060430104612
---------------

OOF RMSLE = 0.7767242727007482

```
# まとめ
フォーラムで紹介されていたOOFを活用したアンサンブル手法をベースに、モデル選定とパラメーターチューニングを行いました。最終評価までの過程では、LB（Leaderboard）も400位台でしたが、多重共線性やLBへのオーバーフィットに注意しながら、LBとの乖離を抑えることに注力。その結果、自身のCV Score（交差検証スコア）を向上させたことが最終評価のスコアに繋がったと考えています。

今回のアプローチでは、すべて同一特徴量を用いてアンサンブルを行いましたが、各モデルごとに最適な特徴量やパラメーターチューニングを追求できれば、更なる精度向上が可能だったと感じています。

メダル圏内に届く結果は得られませんでしたが、今回のコンペを通じて、自身の分析力向上の可能性を大いに実感しました。これからも勉強を続け、より良いモデル構築と分析を目指していきたいと思います。
