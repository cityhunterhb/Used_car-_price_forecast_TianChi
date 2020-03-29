# step 1 导入函数工具箱

## 基础工具
import pandas as pd
import numpy as np
import warnings 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn
from IPython.display import display, clear_output
import time 

warnings.filterwarnings('ignore')
#%matplotlib inline

## 模型预测试
from sklearn import linear_model, preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

## 数据降维处理
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, SparsePCA

import lightgbm as lgb 
import xgboost as xgb 

## 参数搜索和评价
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

###########################################33
#step 2 数据读取
##2 载入数据集和测试集 利用pandas读取数据集
path = './datalab/'
Train_data = pd.read_csv(path + 'used_car_train_20200313.csv', sep=' ')
Test_data = pd.read_csv(path + 'used_car_testA_20200313.csv', sep=' ')

# 输出数据的大小信息
print('Train data shape: ', Train_data.shape)
print('Test data shape:', Test_data.shape)

# Train_data.head() #简要浏览
# Train_data.describe() #统计信息
# Train_data.info() #可看到缺失信息
# Train_data.columns #查看列名

# Test_data.describe();
# Test_data.info()


###########################################################################
#step 3 特征与标签构建
#1 提取数值类型特征列名
numerical_cols = Train_data.select_dtypes(exclude = 'object').columns
print(numerical_cols)

categorical_cols = Train_data.select_dtypes(include='object').columns
print(categorical_cols)

#2 构建训练和测试样本
## 选择特征列
feature_cols = [col for col in numerical_cols if col not in ['SaleID', 'name', 'regDate', 'createDate', 'price', 'model', 'brand', 'regionCode', 'seller']]
feature_cols = [col for col in feature_cols if 'Type' not in col]

## 提取特征列，标签列构造训练样本和测试样本
X_data = Train_data[feature_cols]
Y_data = Train_data['price']

X_test = Test_data[feature_cols]

print('X train shape: ', X_data.shape)
print('X test shape: ', X_test);

## 定义一个统计函数，方便后续信息统计
def Sta_inf(data):
    print('_min', np.min(data))
    print('_max', np.max(data))
    print('_mean', np.mean(data))
    print('_ptp', np.ptp(data))
    print('_std', np.std(data))
    print('_var', np.var(data))


#3 统计标签的基本分布情况
print('Sta of label:')
Sta_inf(Y_data)

## 绘制标签的统计图，查看标签分布
# plt.hist(Y_data)
# plt.show()
# plt.close()


#4 缺省值用-1填补
X_data = X_data.fillna(-1)
X_test = X_test.fillna(-1)


#step 4 模型训练与预测试
#1 利用xgb进行五折交叉验证查看模型的参数结果
## xgb-model
xgr = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, gamma=0, subsample=0.8,\
    colsample_bytree=0.9, max_depth=7)  #, objective='reg:squarederror'
scores_train = []
scores = []

## 5折交叉验证方式
sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_ind, val_ind in sk.split(X_data, Y_data):
    train_x = X_data.iloc[train_ind].values
    train_y= Y_data.iloc[train_ind]
    val_x = X_data.iloc[val_ind].values
    val_y = Y_data.iloc[val_ind]

    xgr.fit(train_x, train_y)
    pred_train_xgb = xgr.predict(train_x)
    pred_xgb = xgr.predict(val_x)

    score_train = mean_absolute_error(train_y, pred_train_xgb)
    scores_train.append(score_train)
    score = mean_absolute_error(val_y, pred_xgb)
    scores.append(score)

# print('Train MAE: ', np.mean(score_train))
# print('Val MAE: ', np.mean(scores))

#2 定义xgb和lbg模型函数
def build_model_xgb(x_train, y_train):
    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, gamma=0, subsample=0.8,\
        colsample_bytree=0.9, max_depth=7) #, objective ='reg:squarederror'
    model.fit(x_train, y_train)
    return model

def build_model_lgb(x_train, y_train):
    estimator = lgb.LGBMRegressor(num_leaves=127, n_estimators=150)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
    }
    gbm = GridSearchCV(estimator, param_grid)
    gbm.fit(x_train, y_train)
    return gbm

#3 切分数据集(Train, Val)进行模型训练，评价和预测
## Split data with val
x_train, x_val, y_train, y_val = train_test_split(X_data, Y_data, test_size=0.3)
print("==============================")
print('Train lgb......')
model_lgb = build_model_lgb(x_train, y_train)
val_lgb = model_lgb.predict(x_val)
MAE_lgb = mean_absolute_error(y_val, val_lgb)
print('MAE of val with lgb: ', MAE_lgb)

print('Predict lgb......')
model_lgb_pre = build_model_lgb(X_data, Y_data)
subA_lgb = model_lgb_pre.predict(X_test)
print('Sta of Predict lgb: ')
Sta_inf(subA_lgb)

print("===========================")
print("Train xgb......")
model_xgb = build_model_xgb(x_train, y_train)
val_xgb = model_xgb.predict(x_val)
MAE_xgb = mean_absolute_error(y_val, val_xgb)
print("MAE of val with xgb: ", MAE_xgb)

print("Predict xgb......")
model_xgb_pre = build_model_xgb(X_data, Y_data)
subA_xgb = model_xgb_pre.predict(X_test)
print("Sta of Predict xgb: ")
Sta_inf(subA_xgb)

#4 进行两个模型的结果加权融合
## 先采取简单加权融合的方式
val_Weighted = (1 - MAE_lgb / (MAE_lgb + MAE_xgb)) * val_lgb + \
    (1 - MAE_xgb / (MAE_xgb + MAE_lgb)) *val_xgb
val_Weighted[val_Weighted < 0] = 10 # 由于我们发现预测的最小值有负数，而真实情况下，price为负是不存在的，由此我们进行对应的后修正
print("MAE of val with Weighted ensemble: ", mean_absolute_error(y_val, val_Weighted))

sub_Weighted = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*subA_lgb+(1-MAE_xgb/(MAE_xgb+MAE_lgb))*subA_xgb

## 查看预测值的统计进行
plt.hist(Y_data)
plt.show()
plt.close()

#5 输出结果
sub = pd.DataFrame()
sub['SaleID'] = Test_data.SaleID
sub['price'] = sub_Weighted
sub.to_csv('./sub_Weighted.csv', index=False)


# # 分类指标评价计算
# ## accuracy
# import numpy as np
# from sklearn.metrics import accuracy_score
# y_pred = [0, 1, 0, 1]
# y_true = [0, 1, 1, 1]
# print('ACC: ', accuracy_score(y_true, y_pred))

# ## Precision, Recall, F1-score
# from sklearn import metrics
# y_pred2 = [0, 1, 0, 0]
# y_true2 = [0, 1, 0, 1]
# print('Precision: ', metrics.precision_score(y_true2, y_pred2))
# print('Recall: ', metrics.recall_score(y_true2, y_pred2));
# print('F1-score: ', metrics.f1_score(y_true2, y_pred2))

# ## AUC
# from sklearn.metrics import roc_auc_score
# y_true3 = np.array([0, 0, 1, 1])
# y_scores = np.array([0.1, 0.4, 0.35, 0.8])
# print('AUC score: ', roc_auc_score(y_true3, y_scores)) 


# #回归指标评价计算
# # coding=utf-8
# # MAPE 自己实现
# def mape(y_true, y_pred):
#     return np.mean(np.abs((y_pred - y_true) / y_true))

# y_true4 = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])
# y_pred4 = np.array([1.0, 4.5, 3.8, 3.2, 3.0, 4.8, -2.2])

# #MSE
# #均方误差 Mean Squared Error 
# print("MSE-Mean Squared Error: ", metrics.mean_squared_error(y_true4, y_pred4))

# #RMSE
# #均方根误差 Root Mean Squared Error
# print("RMSE-Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_true4, y_pred4)));

# #MAE
# #平均绝对误差 Mean Absolute Error
# print("MAE-Mean Absolute Error: ", metrics.mean_absolute_error(y_true4, y_pred4))

# #MAPE
# #平均绝对百分比误差（Mean Absolute Percentage Error）
# print("MAPE-Mean Absolute Percentage Error: ", mape(y_true4, y_pred4));


# ## R2-score
# from sklearn.metrics import r2_score
# y_true5 = [3, -0.5, 2, 7]
# y_pred5 = [2.5, 0.0, 2, 8]
# print("R2-score: ", r2_score(y_true5, y_pred5))