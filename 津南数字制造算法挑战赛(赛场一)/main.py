# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 21:51:04 2019

@author: 705family
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 16:31:31 2019

@author: 705family
"""

import numpy as np 
import pandas as pd 
import lightgbm as lgb
#import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold#,RepeatedKFold
#from sklearn.preprocessing import OneHotEncoder
#from scipy import sparse
import warnings
import re
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

#import matplotlib.pyplot as plt
#import seaborn as sns
#plt.rcParams['font.sans-serif'] = ['SimHei'] #解决中文显示
#plt.rcParams['axes.unicode_minus'] = False #解决符号无法显示

# In[]静态常量
#文件目录
RAW_DATA_PATH = 'data/'
#输入数据文件名
TRAIN = 'jinnan_round1_train_20181227'
TESTA = 'jinnan_round1_testA_20181227'
TESTB = 'jinnan_round1_testB_20190121'
TESTC = 'jinnan_round1_test_20190201'
ANSA = 'jinnan_round1_ansA_20190125'
ANSB = 'jinnan_round1_ansB_20190125'
ANSC = 'jinnan_round1_ans_20190201'
OPTIMIZE = 'optimize'
FUSAI = 'FuSai'
#输出文件名
OUT_PREDICT = 'submit_FuSai'
OUT_OPTIMIZE = 'submit_optimize'
#exception
time_duration_exception = set()
time_point_exception = set()
#Type_Error = [] #数据类型不一致错误样本集

# In[]输入输出接口
def ReadFile(file_name,Type):
    if Type == 'normal':
        data = pd.read_csv(RAW_DATA_PATH + file_name + '.csv',encoding = 'gb18030')
    elif Type == 'ans':
        data = pd.read_csv(RAW_DATA_PATH + file_name + '.csv', names=['样本id','收率'], encoding = 'gb18030')
    else:
        raise Exception("未知读取类型！", Type)
    return data

# In[]数据清洗函数
#def TypeErrorDetecction(data,features):
#    #类型异常检测：如果摸个数据与该列特征的dtype不一致，替换为np.nan
#    #这个函数不对，dtype检测整个特征列，如果有多重类型，则dtype为object,因此该函数无法发挥作用
#    for i in features:
#        for j in range(len(data)):
#            x = data.loc[j,i]
#            try:
#                dataType = str(type(x)).split("'")[1].split(".")[1]
#            except:
#                dataType = str(type(x)).split("'")[1]
#            if dataType != str(data[i].dtype):
#                Type_Error.append([data.loc[j,'样本id'],i,x,str(type(x)),str(data[i].dtype)])
#                data.loc[j,i] = np.nan
#    return data

def DataPre(data):
    #不一定有用的修正
    data.loc[data['样本id']=='sample_1229','A5'] = '22:00:00'
    #B14异常值
    #测试集A，这一个异常值修改后上升巨大，应考虑B14的异常检测，用-1填充
#    data.loc[data['样本id']=='sample_316','B14'] = -1 #原785，怀疑应该是385
    #训练集异常，对结果影响不大，训练集的异常值好像课题提高模型的鲁棒性，测试集的异常值才比较致命
#    data.loc[data['样本id']=='sample_182','B14'] = 400 #原40，怀疑应该是400
#    data.loc[data['样本id']=='sample_124','B14'] = 400 #原40，怀疑应该是400
    #C榜的B14有个值时320，不一定是异常值，但在训练集没出现过，收率也不符合分布规律
#    data.loc[data['样本id']=='sample_443','B14'] = 120 #原40，怀疑应该是400
    #样本id的处理方式
    data.drop(['样本id'], axis=1, inplace=True)
#    data['样本id'] = data['样本id'].apply(lambda x: int(x.split('_')[1]))
    return data

def TrainClean(data):
    #详见A1、A2、A3、A4的特征构造
    data.loc[data['样本id']=='sample_314','A1'] = 300
    #这个异常不改，无法正常运行，因为：30：:00可以正常分解，但int()不能输入空值，之后要考虑这种异常预防
    data.loc[data['样本id']=='sample_130','A11'] = '00:30:00'
    #A25中包含1900异常值，数据类型为Object,构造特征时会出错
    data.loc[data['A25'] == '1900/3/10 0:00', 'A25'] = data['A25'].value_counts().values[0]
    data['A25'] = data['A25'].astype(int)
    #不一定有用的修正
    data.loc[data['样本id']=='sample_496']['A9'] = '7:00:00'
    data.loc[data['样本id']=='sample_1106']['B4'] = '15:00-16:00'
    data.loc[data['样本id']=='sample_969']['B4'] = '19:00-20:05'
    #将几个重要的样本幅值多遍
    #742,1289,266,1926,1181,1832,1824,1767,152,1208,1742
    important = data[data['样本id']=='sample_266']
    for i in range(15):
        data = data.append(important,ignore_index=True)
#    tab = ['sample_742','sample_1289','sample_266','sample_1926','sample_1181',
#           'sample_1832','sample_1824','sample_1767','sample_152','sample_1208','sample_1742']
    tab = ['sample_266','sample_742','sample_1289','sample_1926']
    for i in tab:
        important = important.append(data[data['样本id']== i],ignore_index=True)
    for i in range(1):
        data = data.append(important,ignore_index=True)
    return data

# In[]特征工程
def getDuration(se,fill=-1):
    try:
        sh,sm,eh,em=re.findall(r"\d+\.?\d*",se)
        if int(sh)>int(eh):
            tm = (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600 + 24  #起始时间大于终止时间，说明结束时间是第二天，所以加24
        else:
            tm = (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600
        return tm
    except:
        time_duration_exception.add(se)
        return fill

def getRelativeDuration(times,timee):
    #计算时间点和时间点的相对时间
    try:
        sh,sm,ss = times.split(":")
    except:
        time_point_exception.add(times)
        return -1
    
    try:
        eh,em,es = timee.split(":")
    except:
        time_point_exception.add(timee)
        return -1
    
    if int(sh)>int(eh):
        RelativeDuration = (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600 + 24
    else:
        RelativeDuration = (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600
    return RelativeDuration

def get_PD_Duration(timeD,timee):
    #以时间段特征结束时间为起始时间，以时间点为借宿时间，计算时间段
    try:
        _,_,sh,sm=re.findall(r"\d+\.?\d*",timeD)
    except:
        time_duration_exception.add(timeD)
        return -1
    try:
        eh,em,_ = timee.split(":")
    except:
        time_point_exception.add(timee)
        return -1
    
    if int(sh)>int(eh):
        RelativeDuration = (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600 + 24
    else:
        RelativeDuration = (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600
    return RelativeDuration

def timeTranSecond(t):
    #时间点转化函数
    try:
        t,m,s=t.split(":")
    except:
        if t=='1900/1/9 7:00':
            return 7*3600/3600
        elif t=='1900/1/1 2:30':
            return (2*3600+30*60)/3600
        else:
            return -1
    
    try:
        tm = (int(t)*3600+int(m)*60+int(s))/3600
    except:
        return (30*60)/3600
    
    return tm
 
def BasicFeatureExtractionProcess(data):
    '''
    时间特征信息
    时间点特征：A5,A7,A9,A11,A14,A16,A24,A2,B5,B7
    时间段特征：A20,A28,B4,B9,B10,B11
    缺失值较多：A7,B10,B11
    '''    
    #A1、A2、A3、A4
    #A2和A3应该是不同浓度的氢氧化钠溶液、A4是水
    #这4列相关性可认为为1，一共只有只有3种方案
    #第一种：A1-200，A3-270，A4-470
    #第二种：A1-250，A3-340，A4-590
    #第三种：A1-300，A2+A3+A4-1105 这种方案占90%以上
    #还有一条数据sample_314是 A1-200，A3-405，A4-700,感觉很有可能是记录错误A1应该为300
    #或者只进行了一组实验，感觉可能性较低
    #因此将其构造成一个特征，但该特征区分度不高，应考虑是否舍弃掉、OneHot、编码
    data[['A1','A2','A3','A4']] = data[['A1','A2','A3','A4']].fillna(0)
    data['A1/(A2+A3+A4)'] = data['A1']/(data['A2']+data['A3']+data['A4'])
#    data.drop(['A1', 'A2', 'A3', 'A4'], axis=1, inplace=True)
    
    #A5为初始计时点，没有太大意义，舍弃掉，留个副本用于后续时间相对值计算
    #A6为初始温度，没有缺失值，保留
    data['time_start_point'] = data['A5']
#    data.drop(['A5'], axis=1, inplace=True)
    data['A5'] = data['A5'].apply(timeTranSecond)
    
    #测温时刻与相应温度、压力特征对 
    #A7-A8 A9-A10 其中A7-A8为可选项，缺失值较多，考虑删除该对特征
    data.drop(['A7','A8'], axis=1, inplace=True)
    data['A9'] = data.apply(lambda x: getRelativeDuration(x['time_start_point'], x['A9']), axis=1)
#    data['A10/A9'] = data['A10']/data['A9']
#    data.drop(['A9','A10'], axis=1, inplace=True)
    
    #计时点水解开始 A11-A12-A13，所以时间段应该是相对于起始时刻A5的 
    #A12、A13可直接保留，其中A13只有两条数据不是0.2，没有区分度，应考虑删除
    data['A11_copy'] = data['A11']
    data['A11'] = data.apply(lambda x: getRelativeDuration(x['time_start_point'], x['A11']),axis=1)
    data.drop(['A13'], axis=1, inplace=True)
    #因为A12和A10相近，都在100摄氏度左右，因此考虑采用相对温度，虽然不在一个工序
    data['A12-A10'] = data['A12']-data['A10']
#    data.drop(['A12'], axis=1, inplace=True)
    
    #A14-A15测温组
    data['A14'] = data.apply(lambda x: getRelativeDuration(x['A11_copy'],x['A14']),axis=1)
#    data['A15/A14'] = data['A15']/data['A14']
    data['A15-A12/A14'] = (data['A15']-data['A12'])/data['A14']
    data.drop(['A14','A11_copy'], axis=1, inplace=True)
    
    #A16、A17、A18测量组，因为A16-A9为固定值，因此该特征没有意义
    #A18仅有一条为0.1，其余全部为0.2，没有区分度，舍弃
    #仅保留A17
    data.drop(['A16','A18'], axis=1, inplace=True)
    data['A17-A15'] = data['A17'] - data['A15']
    
    #盐酸滴加过程工艺特征组A19-A25
    #这里之后可以考虑考虑较为复杂的特征工程
    #A22、A23为滴加前后酸碱度，应该测试一下是原始特征好，还是差值特征好
    #A21、A25分别为滴加前后温度，应该测试一下是原始特征好，还是差值特征好
    #A20为滴加持续时间，那A24滴加完成时间的意义为什么，且A24与A20的结束时间不一致
    #注意：A24要在A20之前转化
    data['A24_copy'] = data['A24']
    data['A24'] = data.apply(lambda x: get_PD_Duration(x['A20'],x['A24']),axis=1)
    data['A20'] = data['A20'].apply(lambda x: getDuration(x))
#    data.drop(['A23'], axis=1, inplace=True)
    
    #脱色保温工序
    data['A26']  = data.apply(lambda x: getRelativeDuration(x['A24_copy'],x['A26']),axis=1)
    data['T8-T7'] = data['A27']-data['A25']
    data['A27_copy'] = data['A27']
    data.drop(['A27','A24_copy'], axis=1, inplace=True)
    
    #分离过程
    data['A28'] = data['A28'].apply(lambda x: getDuration(x))
    
    #神秘物质工序 B1-B6
    #B1为神秘物质，B2为其浓度，B3为滴加后的酸碱度，B4为滴加过程，B5为滴加结束时间，B6为滴加结束时温度
    #思考：首先B1*B2可以得到该物质总质量
    #B3只有一个为3.6,和两个空值，其他均为3.5，没有区分度，舍弃掉
    #注意：B5要在B4之前转化，因为B5的转化用到了B4的原始时间
    data['B1*B2'] = data['B1']*data['B2']
    data['B5_copy'] = data['B5']
    data['B5'] = data.apply(lambda x: get_PD_Duration(x['B4'],x['B5']),axis=1)
    data['B4'] = data['B4'].apply(lambda x: getDuration(x))
    data['T11-T8'] = data['B6'] - data['A27_copy']
    #B7-B8测温
    data['B7'] = data.apply(lambda x: getRelativeDuration(x['B5_copy'],x['B7']),axis=1)
    data['T12-T11'] = data['B8']-data['B6']
    data['(T12-T11)/B7'] = data['T12-T11']/data['B7']
#    data.drop(['B1','B2','B3','A27_copy','B5_copy','B6','B8','B7','T12-T11'], axis=1, inplace=True)
    data.drop(['B3','A27_copy','B5_copy'], axis=1, inplace=True)
    
    #最后一道滴加工序B9-B14
    #首先B13是B12的浓度，所以B13和B12可以合并
    #B9-B11为三段滴加过程，其中B10和B11为可选工序，可考虑将三段时间合并
    data['B12*B13'] = data['B12']*data['B13']
    data['B9'] = data['B9'].apply(lambda x: getDuration(x,0))
    data['B10'] = data['B10'].apply(lambda x:getDuration(x,0))
    data['B11'] = data['B11'].apply(lambda x:getDuration(x,0))
    data['B9-B11'] = data['B9']+data['B10']+data['B11']
    data['B12/B14'] = data['B12']/data['B14']
#    data.drop(['B12','B13','B9','B10','B11'], axis=1, inplace=True)
    
    data.drop(['time_start_point'], axis=1, inplace=True)
    #删除温度组
    data.drop(['A12','A15','A17','A21','A25'], axis=1, inplace=True)
    return data

def FeatureEnginearing(data):
    #有风的冬的另一个特征，有点效果，不大，怀疑跟上一个特征相似性比较大 9380-9357
    data['a19/a1_a3_a4_b1_b12_b14'] = data['A19']/(data['A1']+data['A3']+data['A4']+data['B1']+data['B12']+data['B14'])
    #有风的冬提供的强特，B14除其他原料和
    data['b14/a1_a3_a4_a19_b1_b12'] = data['B14']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])
    data.drop(['A1', 'A2', 'A3', 'A4'], axis=1, inplace=True)
    #不在一个工序，B8是温度，B10是时间，不是一个好特征 9392->9380
    data['B8*B10']=data['B8']*data['B10'] 
    #删除没有区分度的特征
#    good_cols = list(data.columns)
#    for col in data.columns:
#        rate = data[col].value_counts(normalize=True, dropna=False).values[0]
#        if rate > 0.9:
#            good_cols.remove(col)
#            print(col,rate)      
#    data = data[good_cols]
    data.drop([ 'A23', 'B2', 'B5', 'B13', 'A1/(A2+A3+A4)'], axis=1, inplace=True)
    return data

def TransFeature(data,test=pd.DataFrame()):
    #这个特征构造测试不行啊
    '''
    1.温度压力等连续值我认为肯定不适合用来转化成类别特征
    留下的时间特征：A5,A9,A11,A20,A24,A26,A28,B4,B7,B9,B10,B11
    较少取值的原料特征：A19,B1(B12,B14)
    '''
    if len(test) > 0:
        data = pd.concat([data,test],axis=0)
    cate_columns = ['A5','A9','A11','A20','A24','A26','A28','B4','B7','B9','B10','B11','A19','B1','B12','B14']
#    cate_columns = data.columns
    for f in cate_columns:
        data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
    if len(test) > 0:
        test = data[-test.shape[0]:]
        data = data[:-test.shape[0]]
    
    data['intTarget'] = pd.cut(data['收率'], 5, labels=False)
    data = pd.get_dummies(data, columns=['intTarget'])
    li = ['intTarget_0','intTarget_1','intTarget_2','intTarget_3','intTarget_4']
    mean_columns = []
    
    for f1 in cate_columns:
        cate_rate = data[f1].value_counts(normalize=True,dropna=False).values[0]
        if cate_rate < 0.90:
            for f2 in li:
                col_name = 'B14_to_'+f1+"_"+f2+'_mean'
                mean_columns.append(col_name)
                order_label = data.groupby([f1])[f2].mean() 
                data[col_name] = data['B14'].map(order_label) #'B14'是类别值，在order_label中查‘B14’中每个样本与f1同值 对应的f2级别的收率比率，属于交叉分箱收率吧，很复杂的特征
                miss_rate = data[col_name].isnull().sum() * 100 / data[col_name].shape[0]
                if miss_rate > 0:
                    data = data.drop([col_name], axis=1)
                    mean_columns.remove(col_name)
                else:
                    if len(test) > 0:
                        test[col_name] = test['B14'].map(order_label)
                    
    data.drop(li, axis=1, inplace=True)
    if len(test) > 0:
        return data,test
    else:
        return data

# In[]集成类
class stacking(object):
    def __init__(self, model_num=2, n_splits=5, n_repeats=2):
        self.model_num = model_num
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.train_result = pd.DataFrame()
    
    def stackDataSet(*args):
        #函数的作用是加一列
        data_stack = args[1]
        if len(args)>2:
            for i in range(len(args)-2):
                i = i+2
                data_stack = np.vstack([data_stack,args[i]])
        else:
            data_stack = data_stack.reshape(1,-1)
        return data_stack.transpose()
    
    def train(self,X_train,y_train):
        #X_train为矩阵，y_train为DataFrame()
        folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590) 
        oof_stack = np.zeros(X_train.shape[0])
        for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(X_train, y_train)):
            print("fold {}".format(fold_))
            trn_data, trn_y = X_train[trn_idx], y_train.iloc[trn_idx].values
            val_data, val_y = X_train[val_idx], y_train.iloc[val_idx].values
            
            clf = BayesianRidge()
            clf.fit(trn_data, trn_y)
            
            oof_stack[val_idx] = clf.predict(val_data)
            #存储模型
            joblib.dump(clf,'stack_model'+str(fold_)+'.pkl')
            
        print("stack score:{:<8.8f}".format(mean_squared_error(y_train.values, oof_stack)/2))
        self.train_result = pd.DataFrame({'real':y_train.values,'pred':oof_stack,'error':(y_train.values-oof_stack)**2})
        
    def predict(self,X_test):
        predictions_stack = np.zeros(len(X_test))
        for i in range(self.n_splits*self.n_repeats):
            clf = joblib.load('stack_model'+str(i)+'.pkl')
            predictions_stack += clf.predict(X_test) / (self.n_splits*self.n_repeats)
        return predictions_stack
    
    def submit(self,X_test,Sample_id,name):
        #输入分别是测试集矩阵、测试集样本id，输出文件名
        pred = self.predict(X_test)
        sub_df = pd.DataFrame()
        sub_df[0] = Sample_id
        sub_df[1] = pred
        sub_df[1] = sub_df[1].apply(lambda x:round(x, 3))
        sub_df.to_csv(name+'.csv', index=False, header=None) 
        
    def evaluate(self,X_test,target):
        pred = self.predict(X_test)
        print("预测均方根误差:{:<8.8f}".format(mean_squared_error(target, pred)/2))
        return pred
    
    def boosting(self,pred_m,evaluate=False,y_train=None):
        #如果要评估，evaluate置为true，并将真实输出放在最后
        pred = (pred_m.sum(axis=1)/np.size(pred_m,1))
        if evaluate:
            print("boosting score:{:<8.8f}".format(mean_squared_error(y_train.values, pred)/2))
        return pred
        
# In[]模型
class lgb_model(object):
    def __init__(self,n_splits=5,trainlen=149,metric='mae',model_name='lgb_model'):
        self.model_name = model_name
        self.metric = metric
        self.param = {'num_leaves': 15,#120, 目前15效果最好
                     'min_data_in_leaf': 10,#30 
                     'objective':'regression',
                     'max_depth': -1,
                     'learning_rate': 0.01,
                     "min_child_samples": 30, #调整这个参数一点变化没有
                     "boosting": "gbdt",
                     "feature_fraction": 0.9,#0.9,
                     "bagging_freq": 1,
                     "bagging_fraction": 0.9,#0.9,
                     "bagging_seed": 11,
                     "metric": self.metric,
                     "lambda_l1": 0.01,#0.1,
#                     "lambda_l2":0.1,
                     "verbosity": -1,
                     "max_bin":50, #试过20,100,50附近比较好,200以上不在变化
                     }
        self.n_splits = n_splits
        self.train_result = pd.DataFrame()
        self.pred_train = np.zeros(trainlen)
    
    def train(self,X_train,y_train):
        folds = KFold(n_splits=5, shuffle=True, random_state=2100)
        oof_lgb = np.zeros(len(X_train))
        
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
            print("fold n°{}".format(fold_+1))
            trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])#,feature_name=feature)
            val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])#,feature_name=feature)
        
            num_round = 10000
            clf = lgb.train(self.param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=False, early_stopping_rounds = 100)
            oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
            #存储模型
            joblib.dump(clf,self.model_name+str(fold_)+'.pkl')
        
        predictions_lgb = np.zeros(len(X_train))
        for i in range(self.n_splits):
            clf = joblib.load(self.model_name+str(i)+'.pkl')
            predictions_lgb += clf.predict(X_train, num_iteration=clf.best_iteration) / self.n_splits
        self.train_result = pd.DataFrame({'real':y_train,'pred':predictions_lgb,'error':(y_train-predictions_lgb)**2})
        
        print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, y_train)/2))
#        self.train_result = pd.DataFrame({'real':y_train,'pred':oof_lgb,'error':(y_train-oof_lgb)**2})
        self.pred_train = oof_lgb
        
    def predict(self,X_test):
        predictions_lgb = np.zeros(len(X_test))
        for i in range(self.n_splits):
            clf = joblib.load(self.model_name+str(i)+'.pkl')
            predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / self.n_splits
        return predictions_lgb
    
    def submit(self,X_test,Sample_id,name):
        #输入分别是测试集矩阵、测试集样本id，输出文件名
        pred = self.predict(X_test)
        sub_df = pd.DataFrame()
        sub_df[0] = Sample_id
        sub_df[1] = pred
        sub_df[1] = sub_df[1].apply(lambda x:round(x, 3))
        sub_df.to_csv(name+'.csv', index=False, header=None)
    
    def evaluate(self,X_test,target):
        pred = self.predict(X_test)
        print("预测均方根误差:{:<8.8f}".format(mean_squared_error(target, pred)/2))
        return pred
    
    def stack_method(self,X,train=True,evaluate=False,y=None,Sample_id=None,name=None):
        #该函数依赖于上面的Stacking类
        #如果要训练，X_train要赋值为True或不给出该参数, 并给出y
        #如果要预测上交，X_train和evaluate必须为False,并给出Sample_id和文件名name
        predictions_lgb = pd.DataFrame()
        for i in range(self.n_splits):
            clf = joblib.load(self.model_name+str(i)+'.pkl')
            predictions_lgb[str(i)] = clf.predict(X, num_iteration=clf.best_iteration)
        stack = stacking()
        if train:
            stack.train(predictions_lgb.values,y) 
        elif evaluate:
            pred = stack.predict(predictions_lgb.values)
            print("repre score: {:<8.8f}".format(mean_squared_error(pred,y.values)/2))
            self.train_result = pd.DataFrame({'real':y.values,'pred':pred,'error':(y.values-pred)**2})
        else:
            pred = stack.predict(predictions_lgb.values)
            sub_df = pd.DataFrame()
            sub_df[0] = Sample_id
            sub_df[1] = pred
            sub_df[1] = sub_df[1].apply(lambda x:round(x, 3))
            sub_df.to_csv(name+'.csv', index=False, header=None)
            
# In[]主流程
#def eval_process():
#    '''
#    #包含异常值的评估流程，因为复赛去除了潜在异常值，所以该流程的调试已经没有了用处
#    #Mr.向提供了一个超级好的思路：因此svm对极端值的预测效果非常好，因此当svm预测值偏离
#    #正常值较大时，采用svm的预测，正常的预测值采用原本的模型
#    #另外发现，绘制预测曲线图能更好的观察到极端值及其预测效果
#    '''
#    #读文件
#    train = ReadFile(TRAIN,'normal')
#    train = TrainClean(train)
#    target = train['收率']
#    del train['收率']
#    testA = ReadFile(TESTA,'normal')
#    testB = ReadFile(TESTB,'normal')
#    testC = ReadFile(TESTC,'normal')
#    ansA = ReadFile(ANSA,'ans')
#    ansB = ReadFile(ANSB,'ans')
#    ansC = ReadFile(ANSC,'ans')
#    #数据清洗、筛选
#    data = pd.concat([train,testA,testB,testC], axis=0, ignore_index=True)
#    data = DataPre(data)
#    #特征工程
#    data = BasicFeatureExtractionProcess(data)
#    data = FeatureEnginearing(data)
#    data = data.fillna(-1)
#    #归一化
##    x_scaler = MinMaxScaler()
##    data[:] = x_scaler.fit_transform(data)
#    
#    train_X = data[:train.shape[0]].values
#    testA = data[train.shape[0]:(train.shape[0]+testA.shape[0])].values
#    testB = data[(train.shape[0]+testA.shape[0]):(train.shape[0]+testA.shape[0]+testB.shape[0])].values
#    testC = data[-testC.shape[0]:].values
#    train_y = target.values
#    #训练并评估
#    model = lgb_model()
#    model.train(train_X,train_y)
#    print('A榜预测结果')
#    ansA['pred'] = model.evaluate(testA,ansA['收率'].values)
#    print('B榜预测结果')
#    ansB['pred'] = model.evaluate(testB,ansB['收率'].values)
#    print('C榜预测结果')
#    ansC['pred'] = model.evaluate(testC,ansC['收率'].values)
#    return ansA, ansB, ansC
            
def eval_process():
    #读文件
    train = ReadFile(TRAIN,'normal')
    testA = ReadFile(TESTA,'normal')
    testB = ReadFile(TESTB,'normal')
    testC = ReadFile(TESTC,'normal')
    ansA = ReadFile(ANSA,'ans')
    ansB = ReadFile(ANSB,'ans')
    ansC = ReadFile(ANSC,'ans')
    #数据清洗、筛选
    train = TrainClean(train)
    target = train['收率']
    del train['收率']
    target = target.append(ansA['收率'],ignore_index=True)
    target = target.append(ansB['收率'],ignore_index=True)
    target = target.append(ansC['收率'],ignore_index=True)
    data = pd.concat([train,testA,testB,testC], axis=0, ignore_index=True)
    data['收率'] = target
    data = data[data['收率']>=0.85] #复赛收率限定在0.85-1之间
    data = data[data['收率']<=1]
    data = data[data['B14']>=350] #B14限定在350-460之间
    data = data[data['B14']<=460]
    
#    target = data['收率']
#    del data['收率']
    sample_id = data['样本id']
    data = DataPre(data)
    #特征工程
    data = BasicFeatureExtractionProcess(data)
    data = FeatureEnginearing(data)
    data = data.fillna(-1)
    
    #验证作死部分
    C = data[-150:]
    y_C = C['收率']
    del C['收率']
    data = data[:-150]
    target = data['收率']
    del data['收率']
    
    train_X = data.values
    train_y = target.values
    #训练并评估
    model = lgb_model(model_name = 'mae_lgb')
    model.train(train_X,train_y)
    
#    model.stack_method(train_X,train=True,y=target)
    print('bagging方式：')
    model.evaluate(C.values,y_C.values)
#    print('stacking方式：')
#    model.stack_method(C.values,train=False,evaluate=True,y=y_C)
        
    model.train_result['样本id'] = pd.Series(sample_id.values)
    model.train_result['B14'] = pd.Series(data['B14'].values)
    model.train_result['B12'] = pd.Series(data['B12'].values)
    
    return model.train_result

def basic_process():
    #读文件
    train = ReadFile(TRAIN,'normal')
    testA = ReadFile(TESTA,'normal')
    testB = ReadFile(TESTB,'normal')
    testC = ReadFile(TESTC,'normal')
    ansA = ReadFile(ANSA,'ans')
    ansB = ReadFile(ANSB,'ans')
    ansC = ReadFile(ANSC,'ans')
    #数据清洗、筛选
    train = TrainClean(train)
    target = train['收率']
    del train['收率']
    target = target.append(ansA['收率'],ignore_index=True)
    target = target.append(ansB['收率'],ignore_index=True)
    target = target.append(ansC['收率'],ignore_index=True)
    data = pd.concat([train,testA,testB,testC], axis=0, ignore_index=True)
    data['收率'] = target
    data = data[data['收率']>=0.85] #复赛收率限定在0.85-1之间
    data = data[data['收率']<=1]
    data = data[data['B14']>=350] #B14限定在350-460之间
    data = data[data['B14']<=460]
    target = data['收率']
    del data['收率']
    sample_id = data['样本id']
    data = DataPre(data)
    #特征工程
    data = BasicFeatureExtractionProcess(data)
    data = FeatureEnginearing(data)
    data = data.fillna(-1)
    #归一化
#    x_scaler = MinMaxScaler()
#    data[:] = x_scaler.fit_transform(data)
    
    train_X = data.values
    train_y = target.values
    #训练并评估
    model = lgb_model(model_name = 'mae_lgb')
    model.train(train_X,train_y)
    
#    model.stack_method(train_X,train=True,y=target)
        
    model.train_result['样本id'] = pd.Series(sample_id.values)
    model.train_result['B14'] = pd.Series(data['B14'].values)
    model.train_result['B12'] = pd.Series(data['B12'].values)
    
#    model2 = lgb_model(metric = 'mse',model_name = 'mse_lgb')
#    model2.train(train_X,train_y)
#    stack = stacking()
#    dataset = stack.stackDataSet(model.pred_train,model2.pred_train)
#    stack.train(dataset,target)
#    stack.boosting(dataset,evaluate=True,y_train = target)
    return model.train_result

#def basic_process():
#    #加入鱼的分类思想的流程，效果一点也不好
#    #读文件
#    train = ReadFile(TRAIN,'normal')
#    testA = ReadFile(TESTA,'normal')
#    testB = ReadFile(TESTB,'normal')
#    testC = ReadFile(TESTC,'normal')
#    ansA = ReadFile(ANSA,'ans')
#    ansB = ReadFile(ANSB,'ans')
#    ansC = ReadFile(ANSC,'ans')
#    #数据清洗、筛选
#    train = TrainClean(train)
#    target = train['收率']
#    del train['收率']
#    target = target.append(ansA['收率'],ignore_index=True)
#    target = target.append(ansB['收率'],ignore_index=True)
#    target = target.append(ansC['收率'],ignore_index=True)
#    data = pd.concat([train,testA,testB,testC], axis=0, ignore_index=True)
#    data['收率'] = target
#    data = data[data['收率']>=0.85] #复赛收率限定在0.85-1之间
#    data = data[data['收率']<=1]
#    data = data[data['B14']>=350] #B14限定在350-460之间
#    data = data[data['B14']<=460]
#    target = data['收率']
#    del data['收率']
#    sample_id = data['样本id']
#    data = DataPre(data)
#    #特征工程
#    data = BasicFeatureExtractionProcess(data)
#    data = FeatureEnginearing(data)
#    data = data.fillna(-1)
#    data['收率'] = target #类别特征作死开始
#    evalData = data[-150:]
#    eval_y = evalData['收率'].values
#    del evalData['收率']
#    data = data[:-150]
#    target = data['收率']
#    data,evalData = TransFeature(data,evalData)
#    eval_X = evalData.values
#    del data['收率']  #作死结束
#    #归一化
##    x_scaler = MinMaxScaler()
##    data[:] = x_scaler.fit_transform(data)
#    
#    train_X = data.values
#    train_y = target.values
#    #训练并评估
#    model = lgb_model()
#    model.train(train_X,train_y)
#    model.train_result['样本id'] = pd.Series(sample_id.values)
#    model.train_result['B14'] = pd.Series(data['B14'].values)
#    model.train_result['B12'] = pd.Series(data['B12'].values)
#    
#    model.evaluate(eval_X,eval_y)
#    return model.train_result

def Optimize_process():
    print('optimize_process')
    optimize = ReadFile(OPTIMIZE,'normal')
    optimize = DataPre(optimize)
    optimize = BasicFeatureExtractionProcess(optimize)
    optimize = FeatureEnginearing(optimize)
    optimize = optimize.fillna(-1)
    basic_process()
    model = lgb_model(model_name = 'mae_lgb')
    result = pd.DataFrame({'result':model.predict(optimize.values)})
    print('预测结果为：{}'.format(result.values))
    result.to_csv(OUT_OPTIMIZE+'.csv', index=False, header=None)  
    
def Submit_process():
    print('submit_process')
    fusai = ReadFile(FUSAI,'normal')
    Sample_id = fusai['样本id']
    fusai = DataPre(fusai)
    fusai = BasicFeatureExtractionProcess(fusai)
    fusai = FeatureEnginearing(fusai)
    fusai = fusai.fillna(-1)
    basic_process()
    model = lgb_model(model_name = 'mae_lgb')
    model.submit(fusai,Sample_id,OUT_PREDICT)

def Search_best_Operation():
    print('find_optimize_process')
    optimize = ReadFile(OPTIMIZE,'normal')
    optimize_raw = optimize.copy()
    optimize = DataPre(optimize)
    optimize = BasicFeatureExtractionProcess(optimize)
    optimize = FeatureEnginearing(optimize)
    model = lgb_model(model_name = 'mae_lgb')
    
    #A28调整没效果
#    fea = 'A28'
#    values = set(np.linspace(0.5,1.5,11))

#    'A26'在0.5以下时最好
#    与A26相关联变量：后面的各个时间都需要提前半小时
#    fea = 'A26'
#    values = set(np.linspace(0,6,61))
#    optimize.loc[0,'A26'] = 0.5

    #A27 T8-T7在-2度的时候最佳，也就是T27应该比T25小2度
    #与T8-T7相关的变量：A27-A25=-2，B6-A27=T11-T8， B8-B6=T12-T11， B8*B10没有办法改
#    fea = 'T8-T7'
#    values = set(np.linspace(-35,30,66))
#    optimize.loc[0,'T8-T7'] = -2
    
    #调整A6没有效果
#    fea = 'A6'
#    values = set(np.linspace(0,100,101))

    #调整A10没有效果
#    fea = 'A10'
#    values = set(np.linspace(95,105,11))
    
    #A19盐酸A20滴加时长
#    fea = 'A19' #190最好-》0.994808
#    values = set(np.linspace(100,350,26))
#    optimize.loc[0,'A19'] = 260
#    fea = 'A20' #0.5-》0.994675
#    values = set(np.linspace(0.1,1,10))
#    optimize.loc[0,'A20'] = 0.5

    #调整A28没有效果
#    fea = 'A28'
#    values = set(np.linspace(0.1,2,20))
    
    #调整B1没有效果,B2效果明显
#    fea = 'B1' 
#    values = range(230,1200,10)
#    fea = 'B2' #0.15-1 ->0.99669432
#    values = set(np.linspace(0.05,3.6,72))
    
    #B13效果拔群，B12、B14没效果
    fea = 'B13' #0.03-》0.99915346
    values = set(np.linspace(0.03,0.15,12))
#    fea = 'B12'
#    values = set(np.linspace(400,1200,8))
#    fea = 'B14'
#    values = set(np.linspace(250,460,21))
    
    tab = pd.DataFrame(columns = [fea,'target'])
    for i in values:
#        optimize.loc[0,fea] = i
        
        optimize = ReadFile(OPTIMIZE,'normal')
        optimize.loc[0,fea] = i
        optimize = DataPre(optimize)
        optimize = BasicFeatureExtractionProcess(optimize)
        optimize = FeatureEnginearing(optimize)
        
        result = model.predict(optimize.values)
        tab = tab.append(pd.DataFrame({fea:i,'target':result}))
    return tab,optimize,optimize_raw
    
if __name__ == "__main__":
    try:
        file_name = sys.argv[1]
        if file_name == "FuSai.csv":
            Submit_process()
        elif file_name == "optimize.csv":
            Optimize_process()
        else:
            print("输入文件名错误！")
    except:
#        ansA,ansB,ansC = eval_process()
#        result = basic_process()
#        sns.pairplot(result, vars=['error', 'real', 'B14','B12'], palette='viridis')
#        plt.figure(figsize=(8,6))
#        plt.scatter(list(result['B14'].values),list(result['error'].values))
#        plt.show()
        tab,optimize,optimize_raw = Search_best_Operation()
#        Optimize_process()
#        result = eval_process()