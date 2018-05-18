# Visualization

## でぃ～ぷら～にんぐとかにやくだつかしかつ～る
じっそうかんきょーはanaconda3-4.2.0 (python3.5)  
chainer:2.0.0  
cupy:1.0.0   
   

### GradCAM
CNNの畳み込み層がどこに注目しているのかを可視化(要chainer)  
[https://qiita.com/nagayosi/items/14f243c058f5a1e7044b](https://github.com/tsurumeso/chainer-grad-cam)

### t-SNE
データの相関を可視化  
`pip install bhtsne`が必要  
[https://github.com/dominiek/python-bhtsne](https://github.com/dominiek/python-bhtsne)

### featureMap
ある畳み込み層の特徴マップをチャンネル数全て可視化(要chainer)(後で載せる)  
[ChainerでCNNしたった - Qiita](https://qiita.com/nagayosi/items/14f243c058f5a1e7044b)

### Generate image by imageMapping.py
Target data is [Office Dataset](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)

<img src="https://github.com/kskdev/Visualization/blob/master/t-SNE/tSNE-office-white-small.png" width="640px">

