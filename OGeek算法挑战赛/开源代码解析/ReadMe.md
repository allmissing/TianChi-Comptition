## 统计特征工程
#### 构造过程
1. 用query_prediction补全prefix生成complete_prefix  
2. 长度相关特征  
   'max_query_ratio'：query_prediction中是否有与title相同的prediction  
   'prefix_word_num'：prefix中包含词汇个数  
   'title_word_num'：title中包含词汇个数  
   'title_len'：title的字节长度  
   'small_query_num'：query_prediction中搜索率小于0.08的prediction个数  
3. 清除prefix、title、complete_prefix中除了数字、字母、汉字以外的字符   
4. 将prefix、title、tag、complete_prefix两两组合连在一起，用于求双字段ctr特征  
5. 求apriori特征：   
   单字段、双字段点击量、搜索量、点击率  
   双字段置信度、支持度  
   
   设计该特征的原因：判断是否点击的本质实际是要看用户搜索意图与title与tag之间是否符合，prefix，query_prediction表征着用户搜索的意图，用于挖掘关联关系的支持度和置信度正好是衡量两者是否有强关联的很好的统计信息。  
   
6. 曝光量特征，例如：在输入相同prefix的前提下，曝光了多少个title  
   求法：但一个字段group，求另一个字段的nunique()  

7. prefix相关特征：  
   "is_in_title"：prefix是否在title 中完整出现  
   "leven_distance"：prefix与title之间的最小编辑距离，用动态规划来求解  
   "distance_rate"：最小编辑距离与prefix、title最大字符串长度的商  
 
8. query_prediction相关特征：  
   prediction中最大搜索率、提供的总搜索率、平均搜索率，提供了几个prediction  
   
## 词向量特征工程
#### 构造过程
1. 用query_prediction补全prefix生成complete_prefix  
2. 清除prefix、title、complete_prefix中除了数字、字母、汉字以外的字符  
3. 计算prefix、title、complete_prefix的句向量prefix_w2v_df、title_w2v_df、complete_prefix_w2v_df  
4. 对句向量进行pca降维到5维  
5. 用句向量聚类，类别数与tags数目差不多，dict存储的是index和预测的聚类类别的对应字典  
6. 用聚类特征来构造单字段点击率特征  
7. prefix、complete_prefix与title句向量的点积  
8. query_prediction与title句向量构造的特征，共三个：  
   prediction中最大搜索率中前三个与title句向量点积相似度的最大值  
   prediction中最大搜索率中前三个与title句向量点积相似度的平均值  
   prediction中最大搜索率中前三个与title句向量点积相似度与搜索率的乘积的和  

#### 做的特征
三类特征：  
1. PCA句向量  
2. 句向量聚类单字段ctr特征  
3. 与title的点积相似度特征  
