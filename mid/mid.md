# 期中作業 --- CNN識別分類物品
* 使用CNN訓練一個可以識別及分類物品的程式。

```
程式為：自己構思，Chatgpt&Claude修改。
```

## 主要功能：
* 自動識別10件物品：
1. 上衣(TOP).
2. 褲子(Trouser).
3. 帽T（Pullover）.
4. 連衣裙（Dress）.
5. 外套(Coat).
6. 涼鞋 (Sandal).
7. 襯衫 (Shirt).
8. 運動鞋 (Sneaker).
9. 包(Bag).
10. 短靴 (Ankle boot).

## 程式各部分：
* 數據處理：
1. 使用Fashion-MNIST 資料集。
2. 隨即水平翻轉圖片。
3. 標準話像素值到[-1,1] 範圍。
4. 只使用5000張圖片進行訓練。

* CNN模型架構：
1. 兩層捲積層提取圖像特徵
2. 批量歸一化加速訓練
3. Dropout 防止過度擬合。
4. 最大池化減少計算量
5. 全連接曾進行最終分類
