# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:22:48 2020

@author: edwin

Function to read running data in Github Repo 'Running'
"""

#Import Essential Packages for loading a csv file into Pandas. 

import pandas as pd

path = 'C:\\Users\\edwin\\OneDrive\\Documents\\Machine_Learning\\Git_Repos\\Running\\Data\\Running_data.csv'
df = pd.read_csv(path)

#Hand chosen 15 columns of interest
Run_data = df[['Distance', 'Date', 'Title', 'Calories','Time', 'Avg HR', 'Max HR', 'Aerobic TE', 'Avg Run Cadence', 'Max Run Cadence', 'Avg Pace', 'Best Pace', 'Elev Gain', 'Elev Loss', 'Avg Stride Length']]


#%% Initial Data QC
def Initial_Data_QC(df): 
    print('\n', 'shape:' , df.shape, '\n')
    print('Columns:', df.columns,'\n' )
    print(df.head(),'\n')
    print(df.tail(),'\n')
    print(df.info(),'\n')
    print(df.describe(),'\n') #Only float columns are shown here
    
#%% Remove unwanted characters from data columns    
    
