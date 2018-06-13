# Visualization Kit for Deep Learning

Deep Learningの分析・可視化とかに利用できるツールをまとめるレポジトリにする予定 <br>
作成理由は可視化手法をいちいち探すのがメンドーだったから

## Environment
Anaconda3-4.2.0 (python3.5.2)
chainer:2.0.0
cupy:1.0.0


### GradCAM
CNNの畳み込み層がどこに注目しているのかを可視化 <br>
[https://qiita.com/nagayosi/items/14f243c058f5a1e7044b](https://github.com/tsurumeso/chainer-grad-cam)


ここに可視化結果を載せる予定

### t-SNE
データの相関を可視化<br>
画像そのものを次元圧縮するスクリプトとネットワークの中間層の特徴ベクトルを抽出し，マッピングするスクリプトの2種類を用意．<br>
`pip install bhtsne` が必要 <br>
[https://github.com/dominiek/python-bhtsne](https://github.com/dominiek/python-bhtsne)


ここに2種類の可視化結果を似せる予定
### 生成結果の例
Target data is [Office Dataset](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)

<img src="https://github.com/kskdev/Visualization/blob/master/t-SNE/tSNE-office-white-small.png" width="640px">



### featureMap
ある畳み込み層の特徴マップをチャンネル数全て可視化(要chainer)(後で載せる) <br>
[ChainerでCNNしたった - Qiita](https://qiita.com/nagayosi/items/14f243c058f5a1e7044b)

ここに可視化結果を載せる予定



### How to define Network

