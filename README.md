# Machine Learning Web App with Titanic

## Overview
Tiatanicデータを使った機械学習を体験することができるウェブアプリケーションである。Trainデータの確認、ヴィジュアライズ、機械学習を使った予測の3部構成で作られている。
### Display trian data
生データと、一部特徴量の基本統計量を確認することができる。
確認したい特徴量はサイドバーから指定できる。また、生データの数は、Num of Rowsから選択できる。
### Visualize features
特徴量をサイドバーから指定すると、指定された特徴量を用いたペアプロットを作成できる。
### Prediction
予測結果はTrain scoreとTest scoreが表示される。シンプルな予測精度と、化学週の程度を確認することができる。
## Discroption Demo
![](images/demo.png)
## 追加予定の機能
* アルゴリズムごとのハイパーパラメータの値を変更できるようにする
* モデルにおける各特徴量の寄与度を確認できるようにする
* 作成した予測をデータベースに保存し、アンサンブルができるようにする
