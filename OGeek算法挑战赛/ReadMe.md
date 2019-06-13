# TIANCHI天池-OGeek算法挑战赛

## 背景
赛题信息链接：https://tianchi.aliyun.com/competition/entrance/231688/information
## 参考
1.鱼的知乎专栏 https://zhuanlan.zhihu.com/p/51422621
## 探索性分析
### 肉眼看数据发现的特点
1. prefix字段包含多种语言、字符，单用中文的词模型来构造词向量应该覆盖率会比较低  
2. prefix和query_prediction是意义对应的，prefix相同，则query_prediction相同  
3. prefix的输入是一个动态过程，所以prefix末尾的词并不一定是个完整的词，但query_prediction是完全包含prefix的，并且prefix位于query_prediction的首位，所以可以有两条思路  
   其一，文章title是否出现在query_prediction中及出现词条对应的搜索率可能是一个强特  
   其二，可以考虑用 query_prediction补全prefix
4. 存在只有tag不同的现象，如第121219-121222的山魈和山鬼，都有百科和音乐的标签，可能是不同的网站，但是相同的名字。
   这里发现一个问题，山魈tag为百科的全部被点击，山鬼tag，为音乐的全部被点击，这里想到一个问题，查山魈的之所以都点百科是因为这个词比较正式、学术，查这个词一般是关注物种，查山鬼的都点击音乐，关于这一点就完全是人的主观了，可以训练一个情感模型来辨别文字的感情色彩
   更深入到该题中就演变成训练一个模型来衡量prefix与tag的匹配度，prefix与title可以用最小编辑距离来衡量，但prefix与tag无法这样做，因为文字差异很大，所以想到训练模型，变成一个多分类问题。
