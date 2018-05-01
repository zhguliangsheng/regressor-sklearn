# -*- coding: utf-8 -*-
"""
这里处理回归问题的一个实例。
"""

import pandas as pd
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.svm import SVR

#这是一个作业。用已知数据预测单耗的值。就是个回归问题。
#数据处理部分删除了一些没有用的数据
#最后整理出了X和y


#数据导入
dataframe=pd.read_excel('./data/processdata.xlsx') #导入excel表 
df0=dataframe.dropna(axis=1)  #删除NAN值所在列 就是删除了备注列
df=df0.set_index(u'日期')     #日期设为索引项
dfx=df.drop([u'单耗',u'衬板总重量（吨）',u'破矿量（吨）',u'衬板时间1#2#月总'],axis=1) #删除列
dfy=df[u'单耗']  #预测项

'''
dfx,dfy是pandas的dataframe格式。
dfx.values相当于特征矩阵 X n*m
dfy.values y n*1矩阵
'''
#前面的不用看了，就是数据整理 
#单耗是需要预测的值 也就是回归项

#预测
X=dfx.values
y=dfy.values
#SVM结果不好，可以不用看了
'''
# SVM回归 SVR
#三个不同的核函数 预测
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)

y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)  
y_poly = svr_poly.fit(X, y).predict(X) #预测值 结果不好
#损失函数 真实值与预测值差的平方 求和 mean_squared_error求损失函数的一个式子 直接调库
svr_rbfcost=mean_squared_error(y,y_rbf)
svr_lincost=mean_squared_error(y,y_lin)
svr_polycost=mean_squared_error(y,y_poly)
'''

#A decision tree is boosted using the AdaBoost
#下面给了三种方法，都是决策树。DecisionTree是标准的决策树模型 深度可以自己指定
#基于AdaBoost（集成方法）和GradientBoosting（梯度提升）的决策树模型 具体的自己搜

regr1_1 = DecisionTreeRegressor(max_depth=4)#深度为4的决策树
regr1_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300)#AdaBoost
regr1_1.fit(X, y)
regr1_2.fit(X, y)
# Predict
y1_1 = regr1_1.predict(X) # 决策树预测值
y1_2 = regr1_2.predict(X) # AdaBoost预测值
#损失函数 上述两个模型的损失函数的值 越小模型预测效果越好 
regr1_1cost=mean_squared_error(y,y1_1) #决策树损失值
regr1_2cost=mean_squared_error(y,y1_2) #AdaBoost损失值

#Gradient Boosting regression
params = {'n_estimators': 500, 'max_depth': 4, 
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X, y)
y_predict=clf.predict(X)  #Gradient Boosting预测值
regr1_3cost=mean_squared_error(y, clf.predict(X))  #Gradient Boosting损失值 
#损失值越小越好 这三个模型里表现最好的是Gradient Boosting
cost=[regr1_1cost,regr1_2cost,regr1_3cost]
print('三个模型的损失值(基于平方损失函数）')
print(cost) #依次输出三个模型的损失值
#plot
#figure1
fig1=plt.figure()

#ax1
#fig1包含2*2=4个图像 add_subplot(2,2,1)在第一个图像上绘制 
ax1= fig1.add_subplot(2, 2, 1)
ax1.plot(y,"b*-",label="true")           #真实值
ax1.plot(y1_1,"x-",label="DecisionTree") #深度为4的决策树的预测值
ax1.plot(y1_2,"+-",label="AdaBoost")   #
ax1.plot(clf.predict(X),'o-',label="GradientBoosting")
ax1.set_xticklabels(['0','1201','1211','1309','1407','1505','1603','1701']) #x轴为时间（月份） 1201代表2012年1月
ax1.set_xlabel('month')  #x轴名称
ax1.set_ylabel('predict value') #y轴名称
ax1.set_title('model')   #图的标题
plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.) #图例的位置 指定为右上角
#ax2
ax2=fig1.add_subplot(2,2,2) #
ax2.plot(y,"b*-",label='true')
ax2.plot(y1_1,"x-",label='DecisionTree')
ax2.set_xticklabels(['0','1201','1211','1309','1407','1505','1603','1701']) #x轴为时间（月份） 1201代表2012年1月
ax2.set_xlabel('month')  #x轴名称
ax2.set_ylabel('predict value') #y轴名称
ax2.set_title('DecisionTree method')
plt.legend(bbox_to_anchor=(1.0, 1),loc=1, borderaxespad=0.)
#ax3
ax3=fig1.add_subplot(2,2,3)
ax3.plot(y,"b*-",label='true')
ax3.plot(y1_2,"+-",label='AdaBoost')
ax3.set_xticklabels(['0','1201','1211','1309','1407','1505','1603','1701']) #x轴为时间（月份） 1201代表2012年1月
ax3.set_xlabel('month')  #x轴名称
ax3.set_ylabel('predict value') #y轴名称
ax3.set_title('AdaBoost method') 
plt.legend(bbox_to_anchor=(1.0, 1),loc=1, borderaxespad=0.) 
#ax4
ax4=fig1.add_subplot(2,2,4)
ax4.plot(y,"b*-",label='true')
ax4.plot(clf.predict(X),"o-",label='GradientBoosting')
ax4.set_xticklabels(['0','1201','1211','1309','1407','1505','1603','1701']) #x轴为时间（月份） 1201代表2012年1月
ax4.set_xlabel('month')  #x轴名称
ax4.set_ylabel('predict value') #y轴名称
ax4.set_title('GradientBoosting method')
plt.legend(bbox_to_anchor=(1.0, 1),loc=1, borderaxespad=0.) 
#figure2
fig2=plt.figure()
#fig2包含1*1=1 add_subplot(1,1,1)代表在第一幅图像上绘制
ax=fig2.add_subplot(1,1,1)
x_cost=np.arange(3)
ax.bar(x_cost,cost) #柱状图
ax.plot(cost,'r*-') #折线图
ax.set_xticklabels(['','','DecisionTree','','AdaBoost','','GradientBoosting',''])
ax.set_title('mean_squared_error') #模型的平方损失均值柱状图和折线图

            
            
            
#第一幅图像拆成四幅

#first
figure3=plt.figure()

ax1_1= figure3.add_subplot(1, 1, 1)
ax1_1.plot(y,"b*-",label="true")           #真实值
ax1_1.plot(y1_1,"x-",label="DecisionTree") #深度为4的决策树的预测值
ax1_1.plot(y1_2,"+-",label="AdaBoost")   #
ax1_1.plot(clf.predict(X),'o-',label="GradientBoosting")
ax1_1.set_xticklabels(['0','1201','1211','1309','1407','1505','1603','1701']) #x轴为时间（月份） 1201代表2012年1月
ax1_1.set_xlabel('month')  #x轴名称
ax1_1.set_ylabel('predict value') #y轴名称
ax1_1.set_title('model')   #图的标题
plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.) #图例的位置 指定为右上角

#second
figure4=plt.figure()
ax2_1=figure4.add_subplot(1,1,1) 
ax2_1.plot(y,"b*-",label='true')
ax2_1.plot(y1_1,"x-",label='DecisionTree')
ax2_1.set_xticklabels(['0','1201','1211','1309','1407','1505','1603','1701']) #x轴为时间（月份） 1201代表2012年1月
ax2_1.set_xlabel('month')  #x轴名称
ax2_1.set_ylabel('predict value') #y轴名称
ax2_1.set_title('DecisionTree method')
plt.legend(bbox_to_anchor=(1.0, 1),loc=1, borderaxespad=0.)   

#third
figure5=plt.figure()
ax3_1=figure5.add_subplot(1,1,1)
ax3_1.plot(y,"b*-",label='true')
ax3_1.plot(y1_2,"+-",label='AdaBoost')
ax3_1.set_xticklabels(['0','1201','1211','1309','1407','1505','1603','1701']) #x轴为时间（月份） 1201代表2012年1月
ax3_1.set_xlabel('month')  #x轴名称
ax3_1.set_ylabel('predict value') #y轴名称
ax3_1.set_title('AdaBoost method') 
plt.legend(bbox_to_anchor=(1.0, 1),loc=1, borderaxespad=0.)

#fourth
figure6=plt.figure()
ax4_1=figure6.add_subplot(1,1,1)
ax4_1.plot(y,"b*-",label='true')
ax4_1.plot(clf.predict(X),"o-",label='GradientBoosting')
ax4_1.set_xticklabels(['0','1201','1211','1309','1407','1505','1603','1701']) #x轴为时间（月份） 1201代表2012年1月
ax4_1.set_xlabel('month')  #x轴名称
ax4_1.set_ylabel('predict value') #y轴名称
ax4_1.set_title('GradientBoosting method')
plt.legend(bbox_to_anchor=(1.0, 1),loc=1, borderaxespad=0.) 
plt.show()#显示图像 两幅图像 