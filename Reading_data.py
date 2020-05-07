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
#Simple function quickly perform many of the straight forward QC's on a dataframe
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
#Creating 'Delta Day' column. This is the number of days since the first date in the dataframe.  
basedate = Run_data.Date.min()
Run_data['Delta_Day'] = (Run_data['Date']-basedate).dt.days
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

#%%  Functions for plotting data

#--------------------- Function 1 ------------------------
# Cross plots of user defined variables (non strings only) - Scatter Matrix 
from pandas.plotting import scatter_matrix
Run_5k_drop = Run_5k.drop(Run_5k[Run_5k.Avg_HR == 0].index)
attributes1 = [ 'Avg_HR', 'Calories', 'Time_dec_min', 'Avg_Run_Cadence']
scatter_matrix(Run_5k_drop[attributes1])

#--------------------- Function 2 ------------------------
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

#--------------------- Function 3 ------------------------
# Function to Cross-Plot any chosen variable with locations coloured
# Requires dictionary to provide color variable
#a = {'Wokingham':'red','Woodley':'blue','TVP':'green','Other':'orange','Nottingham':'black','Wirral':'purple','Erewash':'yellow'}
#a = {'Monday':'red','Tuesday':'blue','Wednesday':'green','Thursday':'orange','Friday':'black','Saturday':'purple','Sunday':'yellow'}
def Cross_Plotter_Color(df,var1,var2,var3,var3_dict,fig,ax): 
    df = df.drop(df[df.Avg_HR == 0].index) #drops zero values (relies on Avg_HR header being 0)
    s = 30 # Marker size
    m = 'o' # Marker shape
    ec='black' # Marker outline color
    for x,y in var3_dict.items(): # Loops through supplied dictionary to access both the variable and its repective color
        tmp_df = df.loc[(df[var3].str.match(x))]
        ax.scatter(tmp_df[var1],tmp_df[var2],c = y, label = x, s=s, marker=m, edgecolors=ec )
        
    plt.xlabel(var1)
    plt.ylabel(var2)
    ax.legend()
    pass

#--------------------- Function 4 ------------------------
# Function to cross plot any two variables of choice
def Cross_Plotter(df,var1,var2, fig, ax):
    df = df.drop(df[df.Avg_HR == 0].index)
    ax.scatter(df[var1],df[var2] )
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    pass
#--------------------- Function 5 ------------------------
#Function to plot pie plot of any variable
def Pie_plot(df, var,fig,ax):
    var_cnt = df[var].value_counts() # Counts number of each variable for plotting
    var_cnt_dict = dict(var_cnt) # Convert from pandas series to dictionary
    labels = list(var_cnt_dict.keys()) # Creating a list from the dictionary keys
    ax.pie(var_cnt,labels=labels,shadow=True, startangle=90,autopct='%1i%%')
    pass

#--------------------- Function 6 ------------------------
# Function to plot pie chart of any variable - explodes the largest portion of the pie
def Pie_explode(df,var1,fig,ax):
    var_cnt = df[var1].value_counts() # Counts number of each variable for plotting
    var_cnt_dict = dict(var_cnt) # Convert from pandas series to dictionary
    labels = list(var_cnt_dict.keys()) # Creating a list from the dictionary keys
    var_cnt_np = var_cnt.to_numpy() #Creates numpy array of variable counts
    explode = np.zeros(len(var_cnt)) #Initalises array of zeros for explosion array
    ei = np.where(var_cnt_np == np.amax(var_cnt_np)) #Finds where the largest pie slice is
    explode[ei] = 0.2 # add 0.2 to the zero matrix to explode largest segment
    ax.pie(var_cnt,labels=labels,shadow=True, startangle=90,autopct='%1i%%', explode=explode)
    pass
#--------------------- Function 7 ------------------------
# Plot Best fit line of user defined order onto defined figure and axis. 
def Add_BestFitLine(df,x,y,order,fig,ax): 
    X = df[x].to_numpy() # Changing dataframe series to numpy arrays
    Y = df[y].to_numpy()
    Coeff= np.polyfit(X,Y,order) # Calculating coeffecients
    P =  np.poly1d(Coeff) 
    ax.plot(X,P(X),ls='--',color='black') # Plotting function against X
    return Coeff # Returns coeffecients
#%% Creating Plots for Summary - 5k Runs

# --------------------- Figure 1 ---------------------------
#Subplot of Date vs Distance and Date vs Time for opening figure. 
    
figure,(ax1,ax2) = plt.subplots(1,2, figsize=(18,12))
Cross_Plotter(Run_data, 'Date', 'Distance',figure,ax1)
Cross_Plotter(Run_data, 'Date', 'Time_dec_min',figure,ax2)
ax1.set_title('All Runs: Date vs Distance')
ax2.set_title('All Runs: Date vs Time (Decimal Minutes)')
plt.show()

# --------------------- Figure 2 ---------------------------
# Exploding pie plot of locations across all data

fig, ax = plt.subplots( figsize=(8,8))
Pie_explode(Run_data, 'Location', fig, ax)
ax.set_title('Running Locations')

# ----------------------- Figure 3 ---------------------------
#Plotting Color scatter plot of all the different Locations
fig, ax = plt.subplots( figsize=(12,8))
a = {'Wokingham':'red','Woodley':'blue','TVP':'green','Other':'orange','Nottingham':'black','Wirral':'purple','Erewash':'yellow'}
Cross_Plotter_Color(Run_data, 'Date', 'Time_dec_min', 'Location', a, fig, ax)
ax.set_ylim(0,100)
ax.set_title('Date vs Time (decimal minutes) - Location Coloured')

# --------------------- Figure 4 ---------------------------
#Exploding pie plot of days of the week ran across all data

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,12))
Pie_explode(Run_data, 'Day', fig, ax1)
ax1.set_title('Days of the Week')
a = {'Monday':'red','Tuesday':'blue','Wednesday':'green','Thursday':'orange','Friday':'black','Saturday':'purple','Sunday':'yellow'}
Cross_Plotter_Color(Run_data,'Date', 'Time_dec_min', 'Day', a, fig,ax2)
ax2.set_ylim(0,100)
ax2.set_title('Date vs Time (decimal minutes)')

# ----------------------Figure 5 --------------------------

fig, ax = plt.subplots( figsize=(12,8))
a = {'Wokingham':'red','Woodley':'blue','TVP':'green','Other':'orange','Nottingham':'black','Wirral':'purple','Erewash':'yellow'}
Cross_Plotter_Color(Run_5k, 'Date', 'Time_dec_min', 'Location', a, fig, ax)
ax.set_title('Date vs Time (decimal minutes) - Location Coloured')

# ---------------------- Figure 6 -------------------------
print(corr_matrix_5k['Time_dec_min'].sort_values(ascending=False)) #Find linear correlations
#Plot 4 apparent positive correlations
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize = (12,7))
Cross_Plotter(Run_5k, 'Avg_Pace', 'Time_dec_min', fig, ax1)
Cross_Plotter(Run_5k, 'Best_Pace', 'Time_dec_min', fig, ax2)
Cross_Plotter(Run_5k, 'Calories', 'Time_dec_min', fig, ax3)
Cross_Plotter(Run_5k, 'Elev_Loss', 'Time_dec_min', fig, ax4)
fig.suptitle('Four variables cross plotted against Time for 5K Runs', fontsize = 16)
plt.show()

# ---------------------- Figure 7 -------------------------
#Plot 4 correlations either negative or no correlation
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize = (12,7))
Cross_Plotter(Run_5k, 'Avg_HR', 'Time_dec_min', fig, ax1)
Cross_Plotter(Run_5k, 'Max_HR', 'Time_dec_min', fig, ax2)
Cross_Plotter(Run_5k, 'Avg_Run_Cadence', 'Time_dec_min', fig, ax3)
Cross_Plotter(Run_5k, 'Avg_Stride_Length', 'Time_dec_min', fig, ax4)
fig.suptitle('Four variables cross plotted against Time for 5K Runs', fontsize = 16)
plt.show()

#%% --------------------- Figure 8 -------------------------
#Delta Day vs Time - Different order best fit Lines
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize = (18,12))
Cross_Plotter(Run_5k, 'Delta_Day', 'Time_dec_min', fig, ax1)
Cross_Plotter(Run_5k, 'Delta_Day', 'Time_dec_min', fig, ax2)
Cross_Plotter(Run_5k, 'Delta_Day', 'Time_dec_min', fig, ax3)
Cross_Plotter(Run_5k, 'Delta_Day', 'Time_dec_min', fig, ax4)
p1 = Add_BestFitLine(Run_5k, 'Delta_Day', 'Time_dec_min', 1, fig, ax1)
p2 = Add_BestFitLine(Run_5k, 'Delta_Day', 'Time_dec_min', 2, fig, ax2)
p3 = Add_BestFitLine(Run_5k, 'Delta_Day', 'Time_dec_min', 3, fig, ax3)
p20 = Add_BestFitLine(Run_5k, 'Delta_Day', 'Time_dec_min', 20, fig, ax4)
ax1.set_title('Order = 1', fontsize = 12)
ax2.set_title('Order = 2', fontsize = 12)
ax3.set_title('Order = 3', fontsize = 12)
ax4.set_title('Order = 20', fontsize = 12)

#%% Section for calculating and printing general statistics which may be interesting
#Function to change decimal minutes to Minutes and seconds: 
def DecMins_to_MinSecs(dm):
    minute = dm - dm%1
    sec =  round((((dm%1)/100)*60),2)
    return minute+sec

#Function to express a time as a string saying x minutes and y seconds (e.g. 5.57 = 5 mins and 57 secs)
def DecMins_to_MinSecs_str(dm):
    minute = dm - dm%1
    sec =  round((((dm%1)/100)*60),2)
    sec_str = str(sec).split('.')[1]
    Min_str = str(minute).split('.')[0]
    MinSec_Str = Min_str + ' minutes' + ' and ' + sec_str + ' seconds' 
    return MinSec_Str
    
    
#Calculating some general statistics: 
#1. Total 5k running improvement from Slowest to Quickest
print('Difference between slowest and quickest 5k run:' , DecMins_to_MinSecs_str(Run_5k.Time_dec_min.max() - Run_5k.Time_dec_min.min()))

#2. Total 5k running improvement from Slowest to Quickest (discarding La Oliva run)
#Remove La Oliva run
Run_5k_no_Oliva = Run_5k.drop(Run_5k[Run_5k.Orig_Location.str.match('La Oliva')].index)
print('Difference between slowest and quickest 5k run (discard La Oliva):' ,DecMins_to_MinSecs_str(Run_5k_no_Oliva.Time_dec_min.max() - Run_5k_no_Oliva.Time_dec_min.min()))

#3. Total 5k time difference between Earliest Date and Latest Date
#Find times run at earliest and latest dates available
Earliest_time = Run_5k.loc[(Run_5k['Date'] == Run_5k.Date.min())].Time_dec_min
Latest_time = Run_5k.loc[(Run_5k['Date'] == Run_5k.Date.max())].Time_dec_min
Time_diff = Earliest_time.values[0] - Latest_time.values[0]
print('Difference between earliest and latest date:' , DecMins_to_MinSecs_str(Time_diff))

#4. Total 5k time difference between Slowest and Quickest at Wokingham 
Wokingham_Quick = Run_5k.loc[(Run_5k['Location'].str.match('Wokingham'))].Time_dec_min.min()
Wokingham_Slow = Run_5k.loc[(Run_5k['Location'].str.match('Wokingham'))].Time_dec_min.max()
print('Time improvement at Wokingham is:', DecMins_to_MinSecs_str(Wokingham_Slow - Wokingham_Quick))

#5. PB 5k time
print('5k PB is:', DecMins_to_MinSecs_str(Run_5k.Time_dec_min.min()))

#%% Creating Plots for Summary - 10k Runs














