import pandas as pd
from pandas import Series
# from pygame import mixer # Load the required library
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import sweetviz as sv
# import pandas_profiling
# from quickda.explore_data import *
def drop(f):
    df=pd.read_csv(f)
    # df=df.drop(df.columns.to_series()["month":"location"], axis=1)
    # df=df.drop(df.columns.to_series()["c6district":"c6shg"], axis=1)
    return df

def fill():
    #TO BE IMPLEMENTED
    pass

def createHTML(df,feature_cols,target_col):
    """
    Create the HTML reports for the dataset given the names
    Params:
    df: dataframe passed,
    feature_cols: The independent variables (what you input)
    target_col: The dependent vars (what you predict)
    """
    data= df
    ##TODO Set the y_all and target_cols in a way that they get Tk input
    y_all=data.loc[:, target_col[0]:target_col[1]]
    target_col_names = data.loc[:, target_col[0]:target_col[1]].columns.values
    for col in target_col_names:
        X_all = data.loc[:, feature_cols[0]:feature_cols[1]]
        # print(y_all[col])
        X_all[col]=y_all[col]
        # print(X_all[col])
        advert_report=""
        feature_config = sv.FeatureConfig(force_num=col)
        if(len(X_all.columns.values)>50):
            advert_report = sv.analyze(X_all, pairwise_analysis="off", feat_cfg=feature_config, target_feat=col)
        else:
            print("here")
            advert_report = sv.analyze(X_all, pairwise_analysis="on", feat_cfg=feature_config, target_feat=col)

        #display the report
        
        advert_report.show_html('./Reports/Breakdowns/'+col+'.html')
    
    advert_report = sv.analyze(y_all,pairwise_analysis="on")
        #display the report
    advert_report.show_html('./Reports/Breakdowns/Preds'+'.html')
    
def createHTML2(df,feature_cols,target_col):
    """
    Create the HTML reports for the dataset given the names
    Params:
    df: dataframe passed,
    feature_cols: The independent variables (what you input)
    target_col: The dependent vars (what you predict)
    """
    data= df
    data.profile_report().to_file(target_col[0]+".html")
    # explore(data,method="profile")

    


    
