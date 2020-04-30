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
    
#%% Prepare data in every column for visualising and ML    

#Where data is missing, it is replaced with --  We will change these to 0s for the time being. 
Run_data = Run_data.replace('--','0')

#First try and optimise types of all columns
Run_data = Run_data.convert_dtypes()

#Calories Column - Remove any commas and convert to int
Run_data['Calories'] = Run_data['Calories'].str.replace(',','')
Run_data['Calories'] = Run_data['Calories'].astype(int)

#Converting certain columns from string to int or float 
Run_data['Avg HR'] = Run_data['Avg HR'].astype(int)
Run_data['Max HR'] = Run_data['Max HR'].astype(int)
Run_data['Aerobic TE'] = Run_data['Aerobic TE'].astype(float)
Run_data['Avg Run Cadence'] = Run_data['Avg Run Cadence'].astype(int)
Run_data['Max Run Cadence'] = Run_data['Max Run Cadence'].astype(int)
Run_data['Elev Gain'] = Run_data['Elev Gain'].astype(int)
Run_data['Elev Loss'] = Run_data['Elev Loss'].astype(int)

#Removing ':' from Avg and Best pace and replacing with a '.' and change to float
Run_data['Avg Pace'] = Run_data['Avg Pace'].str.replace(':','.')
Run_data['Best Pace'] = Run_data['Best Pace'].str.replace(':','.')
Run_data['Avg Pace'] = Run_data['Avg Pace'].astype(float)
Run_data['Best Pace'] = Run_data['Best Pace'].astype(float)

#Converting Date to Datetime: 
Run_data['Date']= pd.to_datetime(Run_data['Date'],format = '%d/%m/%Y %H:%M')
#Calculate decimal minutes from 'Time' and add new column called 'Time_dec_min'
Run_data['Time_dec_min'] = Run_data['Time'].str.split(':').apply(lambda x: ((int(x[0]) * 3600) + (int(x[1])*60) + int(x[2])/60))

print(Run_data.head())
print(Run_data.dtypes)

#%% Data Insights






