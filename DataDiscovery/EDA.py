#Eda file

import glob
import xlrd
import csv
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import operator
from pathlib import Path

file1=open("Correlations.txt","a")

file2=open("instances.txt","a")

def drop(f):
    df=pd.read_csv(f)
    df=df.drop(df.columns.to_series()["month":"location"], axis=1)
    #leave c6unmet as the last column
    df=df.drop(df.columns.to_series()["c6district":"c6anymethod"], axis=1)
    df=df.drop(df.columns.to_series()["c6homevisit":"c6pneurx"], axis=1)
    return df

def getCorrelations(f,r):
    df=drop(f)

    
    df2 = df[[column for column in df if df[column].count() / len(df) >= r]]
    # use the only the columns that have a >= ratio of present to total 
    for c in df.columns:
        if c not in df2.columns:
            print(c, end=", ")
    print('\n')
    
    df = df2

    individual_features_df = []
    
    for i in range(0, len(df.columns) - 1): # -1 because the last column is unmet
        tmpDf = df[[df.columns[i], 'c6unmet']]
        tmpDf = tmpDf[tmpDf[df.columns[i]] != 0]
        individual_features_df.append(tmpDf)
    
            
    all_correlations = {feature.columns[0]: feature.corr()['c6unmet'][0] for feature in individual_features_df}
    all_correlations = sorted(all_correlations.items(), key=operator.itemgetter(1))

    s="\n"
    g=0

    for (key, value) in all_correlations:
        if abs(value)>0.5:
            s+=" {:>15}: {:>15}".format(key, value)+'\n'
            g+=1
        else:
            # print("Dropping: ",(key,value))
            all_correlations.remove((key,value)) #weed out the weakilings

   
    #if r==0.6:
    file1.write("For filename: "+Path(f).stem+ " at a r value of: "+str(r))
    file1.write(" There are {} strongly correlated (corr>0.5 or -0.5>corr) values with c6unmet:\n{}".format(g, s))
    file2.write(s)
    return (all_correlations,df)

#f=r"C:\Users\Devansh\Desktop\Projects\Gupta\SC UPHMIS CBTS6 Merged 72018.csv"
def csvs():
    files=glob.glob(r"C:\Users\Devansh\Desktop\Projects\Gupta\Data\*.csv")
    #print(files)
    instances=dict()
    #files.pop() #remove the merged file
    dfs=[]
    df=[]
    corr=""
    for f in files:
        for r in range(6,7):
            #csv_from_excel(f)
            #f=r"C:\Users\Devansh\Desktop\Projects\Gupta\SC UPHMIS CBTS6 Merged 82018.csv"
            (corr,df)=getCorrelations(f,r/10)
            for (key,value) in corr:
                if key in instances.keys():
                    instances[key]+=1
                else:
                    instances[key]=1 #init the thing
        dfs.append(df)

            
           
       
        file1.write("For file: "+Path(f).stem+" the number ")
    
    print("Vals: ", instances)
    # file2.write("All strong corr values with values\n"+str(instances)) #if we want to write all vals

    instances={k: v for k, v in sorted(instances.items(), key=lambda item: item[1])[-15:]} #do we want to sort?
    file2.write("Most common("+str(len(instances))+") corr. values for our thing\n"+str(instances)) #if we only want the most common ones
    for i in range(len(files)):
        for r in range(6,7):
            f=files[i]

            df=dfs[i]
            for col in df.columns:
                if col not in instances.keys() and  col!='c6unmet':
                    df.drop([col], axis=1,inplace=True)
            print(df)
            sns.pairplot(data=df,
                            x_vars=df.columns[0:len(df.columns) - 1],
                            y_vars=['c6unmet'],kind="reg",plot_kws=dict(scatter_kws=dict(s=2))).savefig(r"./Corr.Graphs/corr"+Path(f).stem+" "+str(r)+".png")
        
   
if __name__ == '__main__':
    file1.truncate(0)
    file2.truncate(0)
    csvs() # this calls your main function
    
