import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import neural_network,ensemble
from sklearn import isotonic, neighbors, multioutput
from pandas import Series
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
# from pygame import mixer # Load the required library
from statistics import mean
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
import glob
from IPython.display import display
from tabulate import tabulate
from sklearn.model_selection import cross_validate
import numpy as np
# from sklearn.metrics import precision_recall_curve
# from matplotlib import pyplot
# from sklearn.metrics import plot_precision_recall_curve
 
from sklearn.model_selection import learning_curve
 
from sklearn.model_selection import validation_curve
from sklearn.metrics import make_scorer

 
 
# file1=open("/content/drive/My Drive/Gupta/CV.txt","a")
# add these for variety. But they weak: linear_model.LinearRegression(fit_intercept=False),neighbors.KNeighborsRegressor(weights='distance'),
trainers=[RandomForestRegressor(min_impurity_decrease=0.0002,bootstrap=False,n_estimators=300),Lasso(alpha=0.1),ElasticNet(random_state=0),neighbors.KNeighborsRegressor(weights='distance',n_neighbors=5)]
names=["RFR","Lasso","ElasticNet","KNR"]
# trainers=[RandomForestRegressor(min_impurity_decrease=0.0002,bootstrap=False,n_estimators=300)]
# names=["RFR"]
models=[]
 
def drop(f):
    df=pd.read_csv(f)
    df=df.drop(df.columns.to_series()["month":"location"], axis=1)
    df=df.drop(df.columns.to_series()["c6district":"c6shg"], axis=1)
    return df
 
def fill():
    #TO BE IMPLEMENTED
    pass
 
def create_arr(f,rows,columns):
    data= f
    # r=0.6
    
    # data = data[[column for column in data if data[column].count() / len(data) >= r]]
 
    fcool, train="",""
 
    feature_cols = data.loc[:, rows[0]:rows[1]].columns.values
 
    
    target_col = data.loc[:, columns[0]:columns[1]].columns.values
    # print(data.loc[:, 'c6iuddel':'c6pneurx'].columns.values)
    
    # print(feature_cols)
    # print(target_col)
 
    
    w=""
    X_all = data.loc[:, rows[0]:rows[1]]
    l=X_all.filter(feature_cols).mean()
    print(l)
    X_all[feature_cols]=X_all[feature_cols].fillna(value=l.iloc[0])
    # X_all.fillna((X_all.mean()),inplace=True)
    
    y_all=data.loc[:, columns[0]:columns[1]]
    l=y_all.filter(target_col).mean()
    print(l)
    y_all[target_col]=y_all[target_col].fillna(value=l.iloc[0])
    # y_all.fillna((y_all.mean()),inplace=True)
    # print(y_all)
    num_all = data.shape[0] 
    num_train = int(num_all*0.8) # 80% of the data
    num_test = num_all - num_train
 
    train_sizes=[step*num_all//10 for step in range(1,8)]
    print(train_sizes)
    
    #print(X_all)
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=num_test, random_state=5)
    y_max=0
    #print( X_test)
    print("Shuffling of data into test and training sets complete!")
    print("Training set: {} samples".format(X_train.shape[0]))
    print("Test set: {} samples".format(X_test.shape[0]))
    y_testl=y_test.to_numpy().tolist()
    i=0
    plt.title("Learning Curve")
    plt.clf()

    for gnb in trainers:
      file1=open("./Reports/LearningCurves/"+names[i]+".txt","a")
      file1.truncate(0)
      
      #X = np.asarray(feature_cols)
      #y = np.asarray(target_col)
      X = np.asarray(X_all)
      y = np.asarray(y_all)
      cv = cross_validate(gnb, X,y, cv=10,scoring='r2',return_train_score='True')
 
      #Learning Curve training
      train_sizes, train_scores, validation_scores = learning_curve(estimator = gnb,X=X_all ,y=y_all , train_sizes = train_sizes, cv = 10,scoring = 'neg_mean_squared_error')
      train_scores_mean = -train_scores.mean(axis = 1)
      validation_scores_mean = -validation_scores.mean(axis = 1)
      plt.style.use('seaborn')
      plt.plot(train_sizes, train_scores_mean, label = 'Training error '+names[i])
      plt.plot(train_sizes, validation_scores_mean, label = 'Validation error '+names[i])
      plt.ylabel('MSE', fontsize = 14)
      plt.xlabel('Training set size', fontsize = 14)
      plt.legend()
      gnb.fit(X_train,y_train)
      # print(str(cv))
      file1.write(str(cv))
      perm = permutation_importance(gnb, X_train, y_train, n_repeats=10,
                                random_state=42)
 
      perm_sorted_idx = perm.importances_mean
      # print(perm_sorted_idx)
      models.append(perm_sorted_idx)
      if(i==0):
        w=gnb.feature_importances_
      
      i+=1
      plt.title('Learning curves for '+names[i-1], fontsize = 18, y = 1.03)
      plt.savefig("./Reports/LearningCurves/LearningCurve"+names[i-1]+".png")
      plt.clf()
 
 
    
    return (w,data.loc[:, rows[0]:rows[1]].columns.values)
 
def main(df,x,y):
    #get the file
    
    (w,k)=create_arr(df,x,y)
 
    file1=open("./Reports/Importances/trained.txt","a")
    file1.truncate(0)
    
    
    feats=dict()
    perms=dict()
    ipx=0
    for key,val in zip(k,w):
        feats[key]=val
        perms[key]=models[0][ipx]
        ipx+=1
    s=""
    fs,ps,useless=[],[],""
    for k1 in list(feats.keys()):
        if k1 in list(perms.keys()):
            if(feats[k1]==0 and perms[k1]==0):
                useless+=k1+"\n"
        s+="Col: "+k1+" Val Feat: "+ str(feats[k1])+ " Val Perm: "+str(perms[k1])+'\n'
        ps.append(perms[k1])
        fs.append(feats[k1])
    
    file2=open("./Reports/Importances/Useless"+".txt","a")
    file2.write(str(useless))
    
    # file1.writelines(s)
    feats=sorted(feats.items(), key=lambda item: item[1])
    perms=sorted(perms.items(), key=lambda item: item[1])
    feats.reverse()
    perms.reverse()
    dat=[]
    s=""
    rank=1
    for i1,i2 in zip(feats,perms):
        dat.append([rank,i1[0]+": "+str(i1[1]),i2[0]+": "+str(i2[1])])
        rank+=1
    print("There: ",fs)
    df=pd.DataFrame(data=dat)
    df.to_csv("./Reports/Importances/RandomForestComplete.csv", encoding='utf-8',index=False)
 
    s+=tabulate(df,showindex=False,headers=["Rank","Tree Based Feature Importance","Permutation based importance"])
    print(s)
    file1.writelines(s)
    i=0
    for permFeats in models:
      file3=open("./Reports/Importances/Permutation"+names[i]+".txt","a")
      file3.truncate(0)
      # file3.write(str(permFeats))
      perms=dict()
      ipx=0
      for key in (k):
          perms[key]=models[0][ipx]
          ipx+=1
      s=""
      ps=[]
      for k1 in list(perms.keys()):          
          ps.append(perms[k1])
      
      perms=sorted(perms.items(), key=lambda item: item[1])
      perms.reverse()
      dat=[]
      s=""
      rank=1
      for i2 in perms:
          dat.append([rank,i2[0]+": "+str(i2[1])])
          rank+=1
      df=pd.DataFrame(data=dat)
      df.to_csv("./Reports/Importances/Importances"+names[i]+'.csv', encoding='utf-8',index=False)
      s+=tabulate(df,showindex=False,headers=["Rank","Permutation based importance"])
      print(s)
      file3.writelines(s)
      i+=1
 
 
 
 
 


