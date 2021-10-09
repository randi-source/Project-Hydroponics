# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 21:08:41 2021

@author: LENOVO
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import io
import pandas_profiling
from plotnine import *
import plotnine


lettuce_raw= pd.read_csv('D:\\Kerja\\Understanding\\Hydroponics\\Wick\\Data_recap_lettuce.csv')
print(lettuce_raw['PPM_AN'])
lettuce_growth_raw=pd.read_csv('D:\\Kerja\\Understanding\\Hydroponics\\Wick\\Data_growth_recap.csv')

print(lettuce_raw.info())
print(lettuce_growth_raw.info())

lettuce_growth_specific=lettuce_growth_raw[(lettuce_growth_raw['Plant_name']=='Selada')]
lettuce_specific=lettuce_raw[(lettuce_raw['Box']==2)]
print(lettuce_specific.head())
print(lettuce_growth_specific.head())

lettuce_specific['Date']=pd.to_datetime(lettuce_specific["Date"])
lettuce_specific.dtypes

print(lettuce_specific.head())

print(lettuce_specific['PPM_AN'].describe())
print(lettuce_specific['PH_AN'].describe())
print(lettuce_specific['TEMP_AN'].describe())
print(lettuce_specific['EC_AN'].describe())

#describing the source water
lettuce_specific_BN=lettuce_specific.where(lettuce_specific['PPM']!= 0)
lettuce_specific_BH=lettuce_specific_BN[pd.notnull(lettuce_specific_BN['PPM'])]
lettuce_specific_BN=lettuce_specific_BN.dropna()
lettuce_specific_BN

print(lettuce_specific_BN['EC'].describe())
print(lettuce_specific_BN['PPM'].describe())
print(lettuce_specific_BN['PH'].describe())

#add linear line for proportional EC, PPM, and pH
lettuce_specific['EC_up']=1200
lettuce_specific['EC_down']=800
lettuce_specific['PH_up']=6.2
lettuce_specific['PH_down']=5.6
print(lettuce_specific)

lettuce_specific_PH=lettuce_specific.where(lettuce_specific['PH_AN']!=0)
lettuce_specific_PH=lettuce_specific_PH[pd.notnull(lettuce_specific_PH['Date'])]
lettuce_specific_PH
lettuce_specific_PH['Date']=pd.to_datetime(lettuce_specific_PH["Date"])
lettuce_specific.dtypes

lettuce_specific.plot(x='Date', y='PPM_AN', figsize=(10,5), grid=True, marker='o', linestyle='-', linewidth=2)
plt.title('Day to Day PPM After Nutrition', loc='center', pad=30, fontsize=20, color='black')
plt.xlabel('Date', fontsize=15)
plt.ylabel('PPM After Nutrition', fontsize=15)
plt.grid(color='darkgray', linestyle=':', linewidth=0.5)
plt.xlim(xmin='2020-12-09', xmax='2021-1-10')
plt.ylim(ymin=0, ymax=1200)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(lettuce_specific_PH['Date'], lettuce_specific_PH['PH_AN'], label='Actual pH', marker='o', linestyle='-', linewidth=2)
plt.plot(lettuce_specific_PH['Date'], lettuce_specific_PH['PH_up'], label='Upper pH limit')
plt.plot(lettuce_specific_PH['Date'], lettuce_specific_PH['PH_down'], label='Lower pH limit')
plt.title('Day to Day pH After Nutrition', loc='center', pad=30, fontsize=20, color='black')
plt.xlabel('Date', fontsize=15)
plt.ylabel('pH After Nutrition', fontsize=15)
plt.grid(color='darkgray', linestyle=':', linewidth=0.5)
plt.ylim(ymin=5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(lettuce_specific['Date'], lettuce_specific['EC_AN'], label='Actual EC',marker='o', linestyle='-', linewidth=2)
plt.plot(lettuce_specific['Date'], lettuce_specific['EC_up'], label='Upper EC limit')
plt.plot(lettuce_specific['Date'], lettuce_specific['EC_down'], label='Lower EC limit')
plt.title('Day to Day Electrical Conductivity After Nutrition', loc='center', pad=30, fontsize=20, color='black')
plt.xlabel('Date', fontsize=15)
plt.ylabel('EC After Nutrition', fontsize=15)
plt.grid(color='darkgray', linestyle=':', linewidth=0.5)
plt.ylim(ymin=750)
plt.tight_layout()
plt.show()

lettuce_specific.plot(x='Date', y='TEMP_AN', figsize=(10,5), grid=True, marker='o', linestyle='-', linewidth=2)
plt.title('Day to Day Water Temperature After Nutrition', loc='center', pad=30, fontsize=20, color='black')
plt.xlabel('Date', fontsize=15)
plt.ylabel('Water Temperature After Nutrition', fontsize=15)
plt.grid(color='darkgray', linestyle=':', linewidth=0.5)
plt.xlim(xmin='2020-12-09', xmax='2021-1-10')
plt.ylim(ymin=5)
plt.tight_layout()
plt.show()

print(lettuce_specific['PPM_AN'].describe())
print(lettuce_specific['PH_AN'].describe())
print(lettuce_specific['TEMP_AN'].describe())
print(lettuce_specific['EC_AN'].describe())


lettuce_specific['Time']=pd.to_datetime(lettuce_specific['Time'])

lettuce_specific['Time']=lettuce_specific['Time'].map(lambda x: x.strftime('%H:%M:%S'))

lettuce_specific_night=lettuce_specific[(lettuce_specific['Time']> '18:00:00')]
print(lettuce_specific_night.head())

lettuce_specific_day=lettuce_specific[(lettuce_specific['Time'] <'18:00:00')]
print(lettuce_specific_day.head())

#Uncontrolable variable at day
print(lettuce_specific_day['Wind_km/h'].describe())
print(lettuce_specific_day['TEMP_AN'].describe())
print(lettuce_specific_day['Humidity_%'].describe())
print(lettuce_specific_day['Wind_quality_aqi'].describe())

#Uncontrolable variable at night
print(lettuce_specific_night['Wind_km/h'].describe())
print(lettuce_specific_night['TEMP_AN'].describe())
print(lettuce_specific_night['Humidity_%'].describe())
print(lettuce_specific_night['Wind_quality_aqi'].describe())

#Uncontrolable variable day & night
print(lettuce_specific['Today_temp_high_celcius'].describe())
print(lettuce_specific['Today_temp_low_celcius'].describe())

#Uncontrolable variable during the day
print(lettuce_specific['Indeks_UV'].describe())
print(lettuce_specific['Pressure_mb'].describe())
print(lettuce_specific['Dew_point_celcius'].describe())
print(lettuce_specific['Precipitation_%'].describe())
print(lettuce_specific['Tutupan_Awan_%'].describe())

lettuce_growth13=lettuce_growth_specific[(lettuce_growth_specific['No_plant']==13)]
lettuce_growth14=lettuce_growth_specific[(lettuce_growth_specific['No_plant']==14)]
lettuce_growth15=lettuce_growth_specific[(lettuce_growth_specific['No_plant']==15)]
lettuce_growth16=lettuce_growth_specific[(lettuce_growth_specific['No_plant']==16)]
lettuce_growth17=lettuce_growth_specific[(lettuce_growth_specific['No_plant']==17)]
lettuce_growth18=lettuce_growth_specific[(lettuce_growth_specific['No_plant']==18)]
lettuce_growth19=lettuce_growth_specific[(lettuce_growth_specific['No_plant']==19)]
lettuce_growth20=lettuce_growth_specific[(lettuce_growth_specific['No_plant']==20)]
lettuce_growth21=lettuce_growth_specific[(lettuce_growth_specific['No_plant']==21)]
lettuce_growth22=lettuce_growth_specific[(lettuce_growth_specific['No_plant']==22)]
lettuce_growth23=lettuce_growth_specific[(lettuce_growth_specific['No_plant']==23)]
lettuce_growth24=lettuce_growth_specific[(lettuce_growth_specific['No_plant']==24)] 

print(lettuce_growth13)
lettuce_growth13['Date']=pd.to_datetime(lettuce_growth13["Date"])
lettuce_growth14['Date']=pd.to_datetime(lettuce_growth14["Date"])
lettuce_growth15['Date']=pd.to_datetime(lettuce_growth15["Date"])
lettuce_growth16['Date']=pd.to_datetime(lettuce_growth16["Date"])
lettuce_growth17['Date']=pd.to_datetime(lettuce_growth17["Date"])
lettuce_growth18['Date']=pd.to_datetime(lettuce_growth18["Date"])
lettuce_growth19['Date']=pd.to_datetime(lettuce_growth19["Date"])
lettuce_growth20['Date']=pd.to_datetime(lettuce_growth20["Date"])
lettuce_growth21['Date']=pd.to_datetime(lettuce_growth21["Date"])
lettuce_growth22['Date']=pd.to_datetime(lettuce_growth22["Date"])
lettuce_growth23['Date']=pd.to_datetime(lettuce_growth23["Date"])
lettuce_growth24['Date']=pd.to_datetime(lettuce_growth24["Date"])
lettuce_growth13.dtypes


print(lettuce_growth13.describe())
print(lettuce_growth14.describe())
print(lettuce_growth15.describe())
print(lettuce_growth16.describe())
print(lettuce_growth17.describe())
print(lettuce_growth18.describe())
print(lettuce_growth19.describe())
print(lettuce_growth20.describe())
print(lettuce_growth21.describe())
print(lettuce_growth22.describe())
print(lettuce_growth23.describe())
print(lettuce_growth24.describe())

lettuce_growth_sorted=lettuce_growth_specific.sort_values(by=['No_plant','Date'], axis=0, ascending=True, inplace=False, kind='quickshort', na_position='last')


print(lettuce_growth_sorted.head(10))


lettuce_growth_sorted['Growth']=np.where(lettuce_growth_sorted['No_plant']==13,lettuce_growth13['Height_cm'].max()-lettuce_growth13['Height_cm'].min(), False)
lettuce_growth_sorted['Growth']=np.where(lettuce_growth_sorted['No_plant']==14,lettuce_growth14['Height_cm'].max()-lettuce_growth14['Height_cm'].min(), False)
lettuce_growth_sorted['Growth']=np.where(lettuce_growth_sorted['No_plant']==15,lettuce_growth15['Height_cm'].max()-lettuce_growth15['Height_cm'].min(), False)
lettuce_growth_sorted['Growth']=np.where(lettuce_growth_sorted['No_plant']==16,lettuce_growth16['Height_cm'].max()-lettuce_growth16['Height_cm'].min(), False)
lettuce_growth_sorted['Growth']=np.where(lettuce_growth_sorted['No_plant']==17,lettuce_growth17['Height_cm'].max()-lettuce_growth17['Height_cm'].min(), False)
lettuce_growth_sorted['Growth']=np.where(lettuce_growth_sorted['No_plant']==18,lettuce_growth18['Height_cm'].max()-lettuce_growth18['Height_cm'].min(), False)
lettuce_growth_sorted['Growth']=np.where(lettuce_growth_sorted['No_plant']==19,lettuce_growth19['Height_cm'].max()-lettuce_growth19['Height_cm'].min(), False)
lettuce_growth_sorted['Growth']=np.where(lettuce_growth_sorted['No_plant']==20,lettuce_growth20['Height_cm'].max()-lettuce_growth20['Height_cm'].min(), False)
lettuce_growth_sorted['Growth']=np.where(lettuce_growth_sorted['No_plant']==21,lettuce_growth21['Height_cm'].max()-lettuce_growth21['Height_cm'].min(), False)
lettuce_growth_sorted['Growth']=np.where(lettuce_growth_sorted['No_plant']==22,lettuce_growth22['Height_cm'].max()-lettuce_growth22['Height_cm'].min(), False)
lettuce_growth_sorted['Growth']=np.where(lettuce_growth_sorted['No_plant']==23,lettuce_growth23['Height_cm'].max()-lettuce_growth23['Height_cm'].min(), False)
lettuce_growth_sorted['Growth']=np.where(lettuce_growth_sorted['No_plant']==24,lettuce_growth24['Height_cm'].max()-lettuce_growth24['Height_cm'].min(), False)


lettuce_growth13['Growth']=lettuce_growth13['Height_cm'].max()-lettuce_growth13['Height_cm'].min()

print(lettuce_growth13['Growth'])
lettuce_growth14['Growth']=lettuce_growth14['Height_cm'].max()-lettuce_growth14['Height_cm'].min()
lettuce_growth15['Growth']=lettuce_growth15['Height_cm'].max()-lettuce_growth15['Height_cm'].min()
lettuce_growth16['Growth']=lettuce_growth16['Height_cm'].max()-lettuce_growth16['Height_cm'].min()
lettuce_growth17['Growth']=lettuce_growth17['Height_cm'].max()-lettuce_growth17['Height_cm'].min()
lettuce_growth18['Growth']=lettuce_growth18['Height_cm'].max()-lettuce_growth18['Height_cm'].min()
lettuce_growth19['Growth']=lettuce_growth19['Height_cm'].max()-lettuce_growth19['Height_cm'].min()
lettuce_growth20['Growth']=lettuce_growth20['Height_cm'].max()-lettuce_growth20['Height_cm'].min()
lettuce_growth21['Growth']=lettuce_growth21['Height_cm'].max()-lettuce_growth21['Height_cm'].min()
lettuce_growth22['Growth']=lettuce_growth22['Height_cm'].max()-lettuce_growth22['Height_cm'].min()
lettuce_growth23['Growth']=lettuce_growth23['Height_cm'].max()-lettuce_growth23['Height_cm'].min()
lettuce_growth24['Growth']=lettuce_growth24['Height_cm'].max()-lettuce_growth24['Height_cm'].min()

lettuce_growth13['Leaf_loose']=lettuce_growth13['Damaged_leaf'].sum()
lettuce_growth14['Leaf_loose']=lettuce_growth14['Damaged_leaf'].sum()
lettuce_growth15['Leaf_loose']=lettuce_growth15['Damaged_leaf'].sum()
lettuce_growth16['Leaf_loose']=lettuce_growth16['Damaged_leaf'].sum()
lettuce_growth17['Leaf_loose']=lettuce_growth17['Damaged_leaf'].sum()
lettuce_growth18['Leaf_loose']=lettuce_growth18['Damaged_leaf'].sum()
lettuce_growth19['Leaf_loose']=lettuce_growth19['Damaged_leaf'].sum()
lettuce_growth20['Leaf_loose']=lettuce_growth20['Damaged_leaf'].sum()
lettuce_growth21['Leaf_loose']=lettuce_growth21['Damaged_leaf'].sum()
lettuce_growth22['Leaf_loose']=lettuce_growth22['Damaged_leaf'].sum()
lettuce_growth23['Leaf_loose']=lettuce_growth23['Damaged_leaf'].sum()
lettuce_growth24['Leaf_loose']=lettuce_growth24['Damaged_leaf'].sum()

lettuce_growth_append=lettuce_growth13.append(lettuce_growth14)
lettuce_growth_append=lettuce_growth_append.append(lettuce_growth15)
lettuce_growth_append=lettuce_growth_append.append(lettuce_growth16)
lettuce_growth_append=lettuce_growth_append.append(lettuce_growth17)
lettuce_growth_append=lettuce_growth_append.append(lettuce_growth18)
lettuce_growth_append=lettuce_growth_append.append(lettuce_growth19)
lettuce_growth_append=lettuce_growth_append.append(lettuce_growth20)
lettuce_growth_append=lettuce_growth_append.append(lettuce_growth21)
lettuce_growth_append=lettuce_growth_append.append(lettuce_growth22)
lettuce_growth_append=lettuce_growth_append.append(lettuce_growth23)
lettuce_growth_append=lettuce_growth_append.append(lettuce_growth24)


print(lettuce_growth_sorted)
lettuce_growth_specific.dtypes

lettuce_growth_specific['Date']=pd.to_datetime(lettuce_growth_specific['Date'])
lettuce_growth_specific.dtypes


lettuce_growth_append['Growth_average']=lettuce_growth_append['Growth']/33

print(lettuce_growth_append)

print(lettuce_growth_append)

lettuce_growth13.describe()

lettuce_growth_append.describe()

lettuce_growth_append.sort_values(by=['Growth'], axis=0, ascending=False, inplace=True, kind='quickshort', na_position='last')
lettuce_growth_append_new=lettuce_growth_append.reset_index()
print(lettuce_growth_append_new)

lettuce_growth_sorted=lettuce_growth_specific.sort_values(by=['No_plant','Date'], axis=0, ascending=True, inplace=False, kind='quickshort', na_position='last')

lettuce_growth_new=lettuce_growth_append_new.drop(['index'], axis=1)

lettuce_growth_new

pd.set_option('display.max_rows', None)
print(lettuce_growth_new)



lettuce_growth13.plot(x='Date', y='Height_cm', label='13', figsize=(10,5), grid=True, marker='o', linestyle='-', linewidth=2)
lettuce_growth14.plot(x='Date', y='Height_cm', label='14',figsize=(10,5), grid=True, marker='o', linestyle='-', linewidth=2)
lettuce_growth15.plot(x='Date', y='Height_cm', label='15',figsize=(10,5), grid=True, marker='o', linestyle='-', linewidth=2)
lettuce_growth16.plot(x='Date', y='Height_cm', label='16',figsize=(10,5), grid=True, marker='o', linestyle='-', linewidth=2)
lettuce_growth17.plot(x='Date', y='Height_cm', label='17',figsize=(10,5), grid=True, marker='o', linestyle='-', linewidth=2)
lettuce_growth18.plot(x='Date', y='Height_cm', label='18',figsize=(10,5), grid=True, marker='o', linestyle='-', linewidth=2)
lettuce_growth19.plot(x='Date', y='Height_cm', label='19',figsize=(10,5), grid=True, marker='o', linestyle='-', linewidth=2)
lettuce_growth20.plot(x='Date', y='Height_cm', label='20',figsize=(10,5), grid=True, marker='o', linestyle='-', linewidth=2)
lettuce_growth21.plot(x='Date', y='Height_cm', label='21',figsize=(10,5), grid=True, marker='o', linestyle='-', linewidth=2)
lettuce_growth22.plot(x='Date', y='Height_cm', label='22',figsize=(10,5), grid=True, marker='o', linestyle='-', linewidth=2)
lettuce_growth23.plot(x='Date', y='Height_cm', label='23',figsize=(10,5), grid=True, marker='o', linestyle='-', linewidth=2)
lettuce_growth24.plot(x='Date', y='Height_cm', label='24',figsize=(10,5), grid=True, marker='o', linestyle='-', linewidth=2)

plt.title('Day to Day Water Temperature After Nutrition', loc='center', pad=30, fontsize=20, color='black')
plt.xlabel('Date', fontsize=15)
plt.ylabel('Water Temperature After Nutrition', fontsize=15)
plt.grid(color='darkgray', linestyle=':', linewidth=0.5)
plt.xlim(xmin='2020-12-09', xmax='2021-1-10')
plt.ylim(ymin=0)
plt.tight_layout()
plt.show()

#the best & worst plant growth without tall posture plant
lettuce_growth17.plot(x='Date', y='Height_cm', label='17',figsize=(10,5), grid=True, marker='o', linestyle='-', linewidth=2)
plt.title('The Best Plant Growth without Tall Posture', loc='center', pad=30, fontsize=20, color='black')
plt.xlabel('Date', fontsize=15)
plt.ylabel('Height Outside Germmination', fontsize=15)
plt.grid(color='darkgray', linestyle=':', linewidth=0.5)
plt.xlim(xmin='2020-12-09', xmax='2021-1-10')
plt.ylim(ymin=0)
plt.tight_layout()
plt.show()


lettuce_growth22.plot(x='Date', y='Height_cm', label='22',figsize=(10,5), grid=True, marker='o', linestyle='-', linewidth=2)
plt.title('The Worst Plant Growth without Tall Posture', loc='center', pad=30, fontsize=20, color='black')
plt.xlabel('Date', fontsize=15)
plt.ylabel('Height Outside Germmination', fontsize=15)
plt.grid(color='darkgray', linestyle=':', linewidth=0.5)
plt.xlim(xmin='2020-12-09', xmax='2021-1-10')
plt.ylim(ymin=0)
plt.tight_layout()
plt.show()

lettuce_growth_new=lettuce_growth_new.sort_values(by=['No_plant','Date'], axis=0, ascending=True, inplace=False, kind='quickshort', na_position='last')







lettuce_growth_new.loc[[15,31,47,63,79,95,111,127,143,159,175,191],'Plants_max']=3

lettuce_growth_new.loc[[0,16,32,48,64,80,96,112,128,144,160,176], 'Plants_min']=1

lettuce_growth_new

lettuce_growth_max=lettuce_growth_new.where(lettuce_growth_new['Plants_max']==3)
print(lettuce_growth_max)
lettuce_growth_max=lettuce_growth_max.drop(['Plants_min'], axis=1)
lettuce_growth_max=lettuce_growth_max.drop(['Plants_max'], axis=1)
lettuce_growth_max=lettuce_growth_max.dropna()
lettuce_growth_max=lettuce_growth_max.reset_index()
print(lettuce_growth_max)

lettuce_growth_min=lettuce_growth_new.where(lettuce_growth_new['Plants_min']==1)
print(lettuce_growth_min)
lettuce_growth_min=lettuce_growth_min.drop(['Plants_max'], axis=1)
lettuce_growth_min=lettuce_growth_min.drop(['Plants_min'], axis=1)
lettuce_growth_min=lettuce_growth_min.dropna()
lettuce_growth_min=lettuce_growth_min.reset_index()
lettuce_growth_min=lettuce_growth_min.drop(['index'], axis=1)
lettuce_growth_min['Height_cm']

print(lettuce_growth_new)
lettuce_growth_new=lettuce_growth_new.drop(['Plants_max'], axis=1)
lettuce_growth_new=lettuce_growth_new.reset_index()
lettuce_growth_new=lettuce_growth_new.drop(['index'], axis=1)

#the best & worst plant leaf damaged during crop cycle

#barchart for plant tall
bars=('13', '14', '15', '16', '17','18','19','20','21','22','23','24')
plt.figure(figsize=(10,7))
plt.bar(bars, lettuce_growth_max['Growth'])
plt.title('Plant Growth of Lettuce', size=15)
plt.xlabel('Plant Numbers', fontsize=10)
plt.ylabel('Plant Growth (cm)', fontsize=10)
plt.tight_layout()
plt.show()

#barchart for plant leaf loose
bars=('13', '14', '15', '16', '17','18','19','20','21','22','23','24')
plt.bar(bars, lettuce_growth_max['Leaf_loose'])
plt.title('Plant Growth of Lettuce', size=14)
plt.xlabel('Plant Numbers', fontsize=15)
plt.ylabel('Plant Total Leaf Loose', fontsize=15)
plt.tight_layout()
plt.show()

#barchart for plant leaf
bars=('13', '14', '15', '16', '17','18','19','20','21','22','23','24')
plt.bar(bars, lettuce_growth_max['Leaf'])
plt.title('Plant Growth of Lettuce', size=14)
plt.xlabel('Plant Numbers', fontsize=15)
plt.ylabel('Plant Total Leaf Loose', fontsize=15)
plt.tight_layout()
plt.show()

#barchart for plant tall start to finish
bars=('13', '14', '15', '16', '17','18','19','20','21','22','23','24')
plt.figure(figsize=(10,7))
plt.bar(bars, lettuce_growth_max['Height_cm'])
plt.bar(bars, lettuce_growth_min['Height_cm'])
plt.title('Plant Growth of Lettuce', size=15)
plt.xlabel('Plant Numbers', fontsize=10)
plt.ylabel('Plant Growth (cm)', fontsize=10)
plt.legend(['Final height', 'Start height'], loc='upper right', prop={'size': 15})
plt.tight_layout()
plt.show()

#barchart for final leaf
bars=('13', '14', '15', '16', '17','18','19','20','21','22','23','24')
plt.figure(figsize=(10, 7))
plt.bar(bars, lettuce_growth_max['Leaf'], color='black')
plt.bar(bars, lettuce_growth_max['Leaf_loose'], alpha=0.5, color='orange')
plt.title('Plant Leaf Quantity of Lettuce', size=15)
plt.xlabel('Plant Numbers', fontsize=10)
plt.ylabel('Plant Leaf', fontsize=10)
plt.legend(['Leaf final', 'Leaf loose'], loc='upper right', prop={'size': 15})
plt.tight_layout()
plt.show()

lettuce_growth_max.columns



#Detailing about the best plant
#change the best plan to 16 and worst plant to 22
print(lettuce_growth18.describe())

lettuce_growth16=lettuce_growth16.reset_index()
lettuce_growth22=lettuce_growth22.reset_index()

lettuce_growth16
lettuce_growth22

lettuce_growth16['Worst_Plant']=lettuce_growth22[['Height_cm']]
print(lettuce_growth17)


plt.figure(figsize=(13,5))
plt.plot(lettuce_growth16['Date'], lettuce_growth16['Height_cm'], label='Best Plant')
plt.plot(lettuce_growth16['Date'], lettuce_growth16['Worst_Plant'], label='Worst Plant')
plt.title('The Best & Worst Plant Growth', loc='center', pad=30, fontsize=20, color='black')
plt.xlabel('Date', fontsize=15)
plt.ylabel('Plant Leaf', fontsize=15)
plt.grid(color='darkgray', linestyle=':', linewidth=0.5)
plt.xlim(xmin='2020-12-09', xmax='2021-01-10')
plt.ylim(ymin=0)
plt.legend('Best Plant','Worst Plant')
plt.tight_layout()
plt.show()


plt.figure(figsize=(13,5))
plt.plot(lettuce_growth16['Date'], lettuce_growth16['Leaf'], label='Leaf Produced')
plt.plot(lettuce_growth16['Date'], lettuce_growth16['Damaged_leaf'], label='Leaf Loss')
plt.title('The Best Plant Leaf Growth', loc='center', pad=30, fontsize=20, color='black')
plt.xlabel('Date', fontsize=15)
plt.ylabel('Plant Leaf', fontsize=15)
plt.grid(color='darkgray', linestyle=':', linewidth=0.5)
plt.xlim(xmin='2020-12-09', xmax='2021-01-10')
plt.ylim(ymin=0)
plt.tight_layout()
plt.show()



plt.figure(figsize=(13,5))
plt.plot(lettuce_growth22['Date'], lettuce_growth22['Leaf'], label='Leaf Produced')
plt.plot(lettuce_growth22['Date'], lettuce_growth22['Damaged_leaf'], label='Leaf Loss')
plt.title('The Worst Plant Leaf Growth', loc='center', pad=30, fontsize=20, color='black')
plt.xlabel('Date', fontsize=15)
plt.ylabel('Plant Leaf', fontsize=15)
plt.grid(color='darkgray', linestyle=':', linewidth=0.5)
plt.xlim(xmin='2020-12-09', xmax='2021-01-10')
plt.ylim(ymin=0)
plt.tight_layout()
plt.show()

lettuce_growth22

#Find the most optimum EC, PPM & pH
#Create new column where the average of plant height

pd.set_option('display.max_rows', None)

lettuce_growth_new=lettuce_growth_new.drop(['Plants_min'], axis=1)
lettuce_growth_new['Height_cm']

lettuce_growth_new['Previous_growth']=lettuce_growth_new['Height_cm'].shift().replace(np.nan, 0).astype(np.int)
lettuce_growth_new['Periodic_growth']=lettuce_growth_new['Height_cm']-lettuce_growth_new['Previous_growth']

lettuce_growth_new

lettuce_growth_new['Periodic_growth']=np.where(lettuce_growth_new['Periodic_growth'] < -1, 0, lettuce_growth_new['Periodic_growth'])

lettuce_growth_new

lettuce_growth_new['Periodic_growth'].describe()

lettuce_growth_date=lettuce_growth_new.where(lettuce_growth_new['Periodic_growth']>2)
lettuce_growth_date=lettuce_growth_date.dropna()
lettuce_growth_date=lettuce_growth_date.sort_values(by=['Date'], axis=0, ascending=True, inplace=False, kind='quickshort', na_position='last')


print(lettuce_growth_date)
lettuce_growth_date['Date']=pd.to_datetime(lettuce_growth_date['Date'])
lettuce_growth_date.dtypes
lettuce_growth_date['Date']=lettuce_growth_date['Date'].map(lambda x: x.strftime('%Y-%m-%d'))


fig, axs = plt.subplots(figsize=(15, 9))
sns.countplot(x='Date', data=lettuce_growth_date )
plt.xlabel('Date', size=15, labelpad=20)
plt.ylabel('Growth Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=15)
plt.xticks(rotation=90)
plt.title('Count of Optimum Growth Period\n In More Than 2 cm/ 2 Day', size=15, y=1.05)
plt.show()

lettuce_growth_25=lettuce_growth_date.where(lettuce_growth_date['Date']=='2020-12-25')
lettuce_growth_3=lettuce_growth_date.where(lettuce_growth_date['Date']=='2021-01-03')

lettuce_growth_25['Periodic_growth'].sum()
lettuce_growth_3['Periodic_growth'].sum()

#the best condition is in 3rd january
#the best plant is no 18 and the worst plant is no 13

print(lettuce_specific.head())

print(lettuce_specific['PPM_AN'].describe())
print(lettuce_specific['PH_AN'].describe())
print(lettuce_specific['TEMP_AN'].describe())
print(lettuce_specific['EC_AN'].describe())

lettuce_specific
lettuce_specific.dtypes
lettuce_specific_best=lettuce_specific.where(lettuce_specific['Date']==datetime.date(2021, 1, 2))
lettuce_specific_best

lettuce_specific.dtypes
lettuce_growth_date.dtypes

lettuce_specific_best=lettuce_specific[(lettuce_specific['Date']=='2021-01-02')]
lettuce_specific_best

lettuce_specific[(lettuce_specific['Date']==datetime.date(2021, 1, 2))]


lettuce_specific_best


print('The best EC to get is',lettuce_specific_best['EC_AN'])
print('The best PPM to get is',lettuce_specific_best['PPM_AN'])
print('The best pH to get is',lettuce_specific_best['PH_AN'])


print('The best Water Temperature to get is',lettuce_specific_best['TEMP_AN'])
print('The best High Temperature to get is',lettuce_specific_best['Today_temp_high_celcius'])
print('The best Low Temperature to get is',lettuce_specific_best['Today_temp_low_celcius'])
print('The best Wind to get is',lettuce_specific_best['Wind_km/h'])
print('The best Humidity to get is',lettuce_specific_best['Humidity_%'])
print('The best Dew Point to get is',lettuce_specific_best['Dew_point_celcius'])
print('The best Pressure to get is',lettuce_specific_best['Pressure_mb'])
print('The best Precipitation to get is',lettuce_specific_best['Precipitation_%'])
print('The best UV Index to get is',lettuce_specific_best['Indeks_UV'])
print('The best Cloud Coverage to get is',lettuce_specific_best['Tutupan_Awan_%'])
print('The best Cloud Height to get is',lettuce_specific_best['Ketinggian_Awan_m'])
print('The best Wind Quality to get is',lettuce_specific_best['Wind_quality_aqi'])






