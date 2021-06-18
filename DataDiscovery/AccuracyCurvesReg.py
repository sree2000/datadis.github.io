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
from sklearn.model_selection import cross_validate,cross_val_score
import numpy as np
# from sklearn.metrics import precision_recall_curve
# from matplotlib import pyplot
# from sklearn.metrics import plot_precision_recall_curve
 
from sklearn.model_selection import learning_curve
 
from sklearn.model_selection import validation_curve
 
from sklearn.metrics import make_scorer
from tabulate import tabulate
from sklearn import model_selection

 
 
trainers=[RandomForestRegressor(min_impurity_decrease=0.0002,bootstrap=False,n_estimators=300),Lasso(alpha=0.1),ElasticNet(random_state=0),neighbors.KNeighborsRegressor(weights='distance',n_neighbors=5)]
names=["RFR","Lasso","ElasticNet","KNR"]
models=[RandomForestRegressor(min_impurity_decrease=0.0002,bootstrap=False,n_estimators=300),Lasso(alpha=0.1),ElasticNet(random_state=0),neighbors.KNeighborsRegressor(weights='distance',n_neighbors=5)]

scoring = ['explained_variance','max_error','neg_mean_absolute_error','neg_mean_squared_error',
'neg_root_mean_squared_error',
'neg_mean_squared_log_error',
'neg_median_absolute_error',
'r2',
'neg_mean_poisson_deviance',
'neg_mean_gamma_deviance']

def error_metrics(i, train_data, train_targ, kfold):
    print(i)
    model=[models[i]]
    error_metrics = pd.DataFrame()
    for scor in scoring:
        score = []
        for mod in model:
          try: 
            result = model_selection.cross_val_score(estimator= mod, X=train_data, y=train_targ,cv=kfold,scoring=scor )
            score.append(result.mean())
          except Exception:
            score.append("Error not Applicable")
            
        error_metrics[scor] =pd.Series(score)
        
    return error_metrics


def drop(f):
    df=pd.read_csv(f)
    df=df.drop(df.columns.to_series()["month":"location"], axis=1)
    df=df.drop(df.columns.to_series()["c6district":"c6shg"], axis=1)
    return df
 
def fill():
    #TO BE IMPLEMENTED
    pass
 

def avg(y_true):
    return (np.mean(y_true)+np.median(y_true)+np.argmax(np.bincount(y_true)))/3

def errFunc(y_true, y_pred):
  return np.mean(1-((sum(y_true)-sum(y_pred))*len(y_pred)/sum(y_true)))

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
    y_all=data.loc[:, columns[0]:columns[1]]

    try:
      l=X_all.filter(feature_cols).mean()
      print(l)
      X_all[feature_cols]=X_all[feature_cols].fillna(value=l.iloc[0])
      # X_all.fillna((X_all.mean()),inplace=True)
      
      l=y_all.filter(target_col).mean()
      print(l)
      y_all[target_col]=y_all[target_col].fillna(value=l.iloc[0])
    except:
      print("Struggling")
    # y_all.fillna((y_all.mean()),inplace=True)
    # print(y_all)
    num_all = data.shape[0] 
    
    train_sizes=[step*num_all//10 for step in range(1,8)]
    print(train_sizes)
    
    #print(X_all)
    
    
    
    my_err = make_scorer(errFunc, greater_is_better=True)
    i=0
    for gnb in trainers:
      file1=open("./Reports/Accuracies/Acc"+names[i]+"Train.txt","a")
      file1.truncate(0)
      scores=[]
      
      for size in train_sizes:
        file1.write("\n\nFor size: "+str(size)+" scores: ")
        num_train = size 
        num_test = num_all - num_train
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=num_test, random_state=5)
        
        X = np.asarray(X_train)
        y = np.asarray(y_train)
        cv = cross_val_score(gnb, X,y, cv=10,scoring=my_err)
        cvTest=cross_val_score(gnb, np.asarray(X_test),np.asarray(y_test), cv=10,scoring=my_err)
        scores.append(np.mean(cv))
        
        
        file1.write(str(np.mean(cv))+ "\n")

        
        # print(str(cv))
        file1.write(str(cv)+"\n\n")

        err=error_metrics(i, X,y, 10)
        file1.write("Trained Metrics w/ size: "+str(size)+"\n")
        file1.write(tabulate(err,headers=scoring)+"\n\n")

        err=error_metrics(i, X_test, y_test, 10)
        # print("14-17", err)
        file1.write("Test Metrics w/ size:"+str(num_test) +"\n")
        file1.write(tabulate(err,headers=scoring))

      plt.style.use('seaborn')
      plt.plot(train_sizes, scores, label = 'Avg Accuracy error' +names[i]+'Train')
      # plt.plot(train_sizes, scores2, label = 'Avg Accuracy error' +names[i]+'Test')

      i+=1
    plt.ylabel('Error')
    plt.xlabel('Training set size')
    plt.legend()
    plt.title('Training Accuracy curves for all')
    plt.savefig("./Reports/Accuracies/AccuracyCurveTrain.png")
    # plt.clf()
 
 
    
    return (w,"")
 
def main(df,x,y):
    #get the file
    
    (w,k)=create_arr(df,x,y)
 
    
 

 
 
 

