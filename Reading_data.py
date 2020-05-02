# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:22:48 2020

@author: edwin

Function to read running data in Github Repo 'Running'
"""

#Import Essential Packages for loading a csv file into Pandas. 

import pandas as pd
import matplotlib.pyplot as plt


path = 'C:\\Users\\edwin\\OneDrive\\Documents\\Machine_Learning\\Git_Repos\\Running\\Data\\Running_data.csv'
df = pd.read_csv(path)

#Hand chosen 15 columns of interest
Run_data = df[['Distance', 'Date', 'Title', 'Calories','Time', 'Avg HR', 'Max HR', 'Aerobic TE', 'Avg Run Cadence', 'Max Run Cadence', 'Avg Pace', 'Best Pace', 'Elev Gain', 'Elev Loss', 'Avg Stride Length']]
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
#Collecting All Locations with less than 4 runs (Birkenhead, Reading, Wallingford, Pembrokeshire, La Oliva, Running?)
Run_data['Location'] = Run_data['Location'].str.replace('Birkenhead Running','Other') 
Run_data['Location'] = Run_data['Location'].str.replace('Reading Parkrun','Other') 
Run_data['Location'] = Run_data['Location'].str.replace('Wallingford 10km','Other') 
Run_data['Location'] = Run_data['Location'].str.replace('Pembrokeshire Running','Other') 
Run_data['Location'] = Run_data['Location'].str.replace('La Oliva Running','Other') 
Run_data['Location'] = Run_data['Location'].str.replace('Running','Other', )
New_Locations = Run_data['Location'].value_counts()

#Converting Date to Datetime: 
Run_data['Date']= pd.to_datetime(Run_data['Date'],format = '%d/%m/%Y %H:%M')

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

#Create Seperate dataframes per Location (if needed)
Run_5k_Wokingham = Run_5k.loc[(Run_5k['Location'].str.match('Wokingham'))]
Run_5k_Woodley = Run_5k.loc[(Run_5k['Location'].str.match('Woodley'))]
Run_5k_TVP = Run_5k.loc[(Run_5k['Location'].str.match('TVP'))]
Run_5k_Other =  Run_5k.loc[(Run_5k['Location'].str.match('Other'))]
Run_5k_Nottingham =  Run_5k.loc[(Run_5k['Location'].str.match('Nottingham'))]
Run_5k_Wirral =  Run_5k.loc[(Run_5k['Location'].str.match('Wirral'))]
Run_5k_Erewash =  Run_5k.loc[(Run_5k['Location'].str.match('Erewash'))]




