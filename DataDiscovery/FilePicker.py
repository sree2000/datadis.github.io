import cv2
import glob
import numpy as np
from tkinter import filedialog
from tkinter import *
import pandas as pd
import Credentials as cred
import HtmlCreator as sv
import os
#pick Video file
#Vid-->imgs
#imgs-->texts
#texts-->imgs
#comparison functions
#new imgs-->vid
    
def main():
    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("data files","*.csv"),("all files","*.*")))
    print(root.filename)
    df=pd.read_csv(root.filename)
    cols=df.columns.values
    print(cols)
    things=cred.getCreds()
    try:  
        os.mkdir("./Breakdowns")
    except :
        pass
    # print("Things: ",things)
    print(things[:2],things[2:])
    sv.createHTML(df,things[:2],things[2:])
    #iv.processImg(r,'jpg')

# main()