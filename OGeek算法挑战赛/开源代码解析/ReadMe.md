# 开源代码解析

## 代码来源


## 词向量特征工程解析

#### 构造过程
1.用query_prediction补全prefix生成complete_prefix  
2.清除prefix、title、complete_prefix中除了数字、字母、汉字以外的字符  
3.计算prefix、title、complete_prefix的句向量prefix_w2v_df、title_w2v_df、complete_prefix_w2v_df  
4.对句向量进行pca降维到5维  
5.用句向量聚类，类别数与tags数目差不多，dict存储的是index和预测的聚类类别的对应字典  
6.用聚类特征来构造单字段点击率特征  
7.prefix、complete_prefix与title句向量的点积  
8.query_prediction与title句向量构造的特征，共三个：  
   prediction中最大搜索率中前三个与title句向量点积相似度的最大值  
   prediction中最大搜索率中前三个与title句向量点积相似度的平均值  
   prediction中最大搜索率中前三个与title句向量点积相似度与搜索率的乘积的和  

#### 做的特征
3类特征：  
1. PCA句向量  
2. 句向量聚类单字段ctr特征  
3. 与title的点积相似度特征  
