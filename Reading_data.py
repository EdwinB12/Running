# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:22:48 2020

@author: edwin

Function to read running data in Github Repo 'Running'
"""

#Importing required modules

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = 'C:\\Users\\edwin\\OneDrive\\Documents\\Machine_Learning\\Git_Repos\\Running\\Data\\Running_data.csv'
Orig_df = pd.read_csv(path) # This saves off original Dataframe incase it is needed later. 
Run_data = pd.read_csv(path) # This is the df we will work with throughoutt the script. 
#Hand chosen 15 columns of interest
Run_data = Run_data[['Distance', 'Date', 'Title', 'Calories','Time', 'Avg HR', 'Max HR', 'Aerobic TE', 'Avg Run Cadence', 'Max Run Cadence', 'Avg Pace', 'Best Pace', 'Elev Gain', 'Elev Loss', 'Avg Stride Length']]
#Modifying and tidying 'Title' Label - Change column name to 'Location'. 
Run_data.rename(columns={'Title': 'Location', 'Avg HR':'Avg_HR', 'Max HR':'Max_HR', 'Aerobic TE':'Aerobic_TE', 'Avg Run Cadence':'Avg_Run_Cadence','Max Run Cadence':'Max_Run_Cadence','Avg Pace':'Avg_Pace', 'Best Pace':'Best_Pace', 'Elev Gain':'Elev_Gain','Elev Loss':'Elev_Loss', 'Avg Stride Length':'Avg_Stride_Length'},inplace=True)

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
Run_data['Avg_HR'] = Run_data['Avg_HR'].astype(int)
Run_data['Max_HR'] = Run_data['Max_HR'].astype(int)
Run_data['Aerobic_TE'] = Run_data['Aerobic_TE'].astype(float)
Run_data['Avg_Run_Cadence'] = Run_data['Avg_Run_Cadence'].astype(int)
Run_data['Max_Run_Cadence'] = Run_data['Max_Run_Cadence'].astype(int)
Run_data['Elev_Gain'] = Run_data['Elev_Gain'].astype(int)
Run_data['Elev_Loss'] = Run_data['Elev_Loss'].astype(int)

#Removing ':' from Avg and Best pace and replacing with a '.' and change to float
Run_data['Avg_Pace'] = Run_data['Avg_Pace'].str.replace(':','.')
Run_data['Best_Pace'] = Run_data['Best_Pace'].str.replace(':','.')
Run_data['Avg_Pace'] = Run_data['Avg_Pace'].astype(float)
Run_data['Best_Pace'] = Run_data['Best_Pace'].astype(float)

#Saving off original 'Title' column before editing
Run_data['Orig_Location'] = Run_data['Location']

#Reduce Location categories, many are repeats with different spellings, as seen below
Orig_Locations = Run_data['Location'].value_counts() #18 categories
#Collect all strings for Wokingham
Run_data['Location'] = Run_data['Location'].str.replace('Wokingham Running','Wokingham')
Run_data['Location'] = Run_data['Location'].str.replace('Wokingham - Running','Wokingham')
Run_data['Location'] = Run_data['Location'].str.replace('Wokingham running','Wokingham')
#Collect all strings for Woodley
Run_data['Location'] = Run_data['Location'].str.replace('Woodley Running','Woodley')
Run_data['Location'] = Run_data['Location'].str.replace('Woodley Other','Woodley')
Run_data['Location'] = Run_data['Location'].str.replace('Woodley running','Woodley')
#Collect all strings for TVP
Run_data['Location'] = Run_data['Location'].str.replace('TVP Running','TVP')
Run_data['Location'] = Run_data['Location'].str.replace('TVP running','TVP')
#Shortening Nottingham and Wirral
Run_data['Location'] = Run_data['Location'].str.replace('Wirral Running','Wirral')
Run_data['Location'] = Run_data['Location'].str.replace('Nottingham Running','Nottingham')
#Collect all strings for Erewash
Run_data['Location'] = Run_data['Location'].str.replace('Erewash Running','Erewash')
Run_data['Location'] = Run_data['Location'].str.replace('Erewash - Running','Erewash')
Loc_count = Run_data['Location'].value_counts() # Counting locations before grouping into 'Other'
#Collecting All Locations with less than 4 runs (Birkenhead, Reading, Wallingford, Pembrokeshire, La Oliva, Running?)
Run_data['Location'] = Run_data['Location'].str.replace('Birkenhead Running','Other') 
Run_data['Location'] = Run_data['Location'].str.replace('Reading Parkrun','Other') 
Run_data['Location'] = Run_data['Location'].str.replace('Wallingford 10km','Other') 
Run_data['Location'] = Run_data['Location'].str.replace('Pembrokeshire Running','Other') 
Run_data['Location'] = Run_data['Location'].str.replace('La Oliva Running','Other') 
Run_data['Location'] = Run_data['Location'].str.replace('Running','Other', )


#Converting Date to Datetime: 
Run_data['Date']= pd.to_datetime(Run_data['Date'],format = '%d/%m/%Y %H:%M')
#Adding day of the week column
Run_data['Day'] = Run_data.Date.dt.day_name()

#Calculate decimal minutes from 'Time' and add new column called 'Time_dec_min'
Run_data['Time_dec_min'] = Run_data['Time'].str.split(':').apply(lambda x: (((int(x[0]) * 3600) + (int(x[1])*60) + int(x[2]))/60))

#Convert Original 'Time' header to a float
Run_data['Time'] = Run_data['Time'].str.replace(':','')
Run_data['Time'] = Run_data['Time'].astype(float)


#%% Splitting Data

#Create seperate dataframes for 5k runs, more than 5k and less than 5k. 
Run_5k = Run_data.loc[(Run_data['Distance'] > 4.9) & (Run_data['Distance'] < 5.1)]
Run_5k_less = Run_data.loc[(Run_data['Distance'] < 4.9)]
Run_5k_more = Run_data.loc[(Run_data['Distance'] > 5.1)]
Run_10k = Run_data.loc[(Run_data['Distance'] > 9.9) & (Run_data['Distance'] < 10.1)]

#Create Seperate dataframes per Location (if needed). This should be done via function!
Run_5k_Wokingham = Run_5k.loc[(Run_5k['Location'].str.match('Wokingham'))]
Run_5k_Woodley = Run_5k.loc[(Run_5k['Location'].str.match('Woodley'))]
Run_5k_TVP = Run_5k.loc[(Run_5k['Location'].str.match('TVP'))]
Run_5k_Other =  Run_5k.loc[(Run_5k['Location'].str.match('Other'))]
Run_5k_Nottingham =  Run_5k.loc[(Run_5k['Location'].str.match('Nottingham'))]
Run_5k_Wirral =  Run_5k.loc[(Run_5k['Location'].str.match('Wirral'))]
Run_5k_Erewash =  Run_5k.loc[(Run_5k['Location'].str.match('Erewash'))]

#%%  Data Visualisation and Insights

#--------------------- Figure 1 ------------------------
# Cross plots of user defined variables (non strings only) - Scatter Matrix 
from pandas.plotting import scatter_matrix
Run_5k_drop = Run_5k.drop(Run_5k[Run_5k.Avg_HR == 0].index)
attributes1 = [ 'Avg_HR', 'Calories', 'Time_dec_min', 'Avg_Run_Cadence']
scatter_matrix(Run_5k_drop[attributes1])

#--------------------- Figure 2 ------------------------
# Correlation Matrix - Best to view in Variable Explorer
corr_matrix = Run_data.corr() 
corr_matrix_5k = Run_5k.corr()
import seaborn as sns
plt.figure()
sns.heatmap(corr_matrix, center=0, cmap='magma', vmin=-1, vmax=1)
plt.title('Correlation Matrix for All Runs')
plt.figure()
sns.heatmap(corr_matrix_5k, center=0, cmap='magma', vmin=-1, vmax=1)
plt.title('Correlation Matrix for All 5K Runs')


#--------------------- Figure 3 ------------------------
# Function to Cross-Plot any chosen variable with locations coloured
# Requires dictionary to provide location and respective color
a = {'Wokingham':'red','Woodley':'blue','TVP':'green','Other':'orange','Nottingham':'black','Wirral':'purple','Erewash':'yellow'}

def Cross_Plotter_Color(df,var1,var2,Loc_dict): 
    df = df.drop(df[df.Avg_HR == 0].index) #drops zero values (relies on Avg_HR header being 0)
    fig,ax = plt.subplots(figsize=(16,12))
    s = 30 # Marker size
    m = 'o' # Marker shape
    ec='black' # Marker outline color
    for x,y in Loc_dict.items():
        tmp_df = df.loc[(df['Location'].str.match(x))]
        ax.scatter(tmp_df[var1],tmp_df[var2],c = y, label = x, s=s, marker=m, edgecolors=ec )
        
    plt.xlabel(var1)
    plt.ylabel(var2)
    ax.legend()
    plt.show()

#--------------------- Figure 4 ------------------------
# Function to cross plot any two variables of choice
def Cross_Plotter(df,var1,var2):
    df = df.drop(df[df.Avg_HR == 0].index)
    fig,ax = plt.subplots(figsize=(12,8))
    ax.scatter(df[var1],df[var2])
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.show()

#--------------------- Figure 5 ------------------------
# Function to plot pie chart of days of the week spent running
    
def Day_Pie(df):
    day_cnt = df.Day.value_counts() # Count number of each day
    day_cnt_dict = dict(day_cnt) 
    labels = list(day_cnt_dict.keys()) # Creating a list from the dictionary keys
    fig,ax = plt.subplots(figsize=(8,8))
    ax.pie(day_cnt,labels=labels,shadow=True, startangle=90,autopct='%1i%%')
    plt.show()

#--------------------- Figure 6 ------------------------
# Function to plot pie chart of days of the week spent running

def Day_Pie_explode(df):
    day_cnt = df.Day.value_counts() # Count number of each day
    day_cnt_dict = dict(day_cnt) 
    labels = list(day_cnt_dict.keys()) # Creating a list from the dictionary keys
    day_cnt_np = day_cnt.to_numpy()
    explode = np.zeros(len(day_cnt))
    ei = np.where(day_cnt_np == np.amax(day_cnt_np))
    explode[ei] = 0.2
    #explode=(0.2,0,0,0,0,0,0) 
    fig,ax = plt.subplots(figsize=(8,8))
    ax.pie(day_cnt,labels=labels,shadow=True, startangle=90,autopct='%1i%%', explode=explode)
    #plt.title('Days of the Week - All Runs')
    plt.show()



