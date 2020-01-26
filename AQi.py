# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 21:01:17 2020

@author: Admin
"""

import requests
from bs4 import BeautifulSoup
from csv import writer
import pandas as pd
import numpy as np



    
with open('indep.csv','w') as file:
    csv_writer = writer(file)
    csv_writer.writerow(['T','TM','tm','H','P','VV','V','VM'])

    for year in range(2013,2019):
        for month in range(1,13):
            if month<10:
                url = 'https://en.tutiempo.net/climate/0{}-{}/ws-430030.html'.format(month,year)
            else:
                url = 'https://en.tutiempo.net/climate/{}-{}/ws-430030.html'.format(month,year)
            
            response = requests.get(url).text.encode('utf=8')                       
            soup = BeautifulSoup(response,'html.parser')
            per_day = soup.find_all(class_='medias mensuales numspan')        
            
            for tbody in per_day:
                for i in range (1,32):                    
                    if tbody.find_all('tr')[i].find_all('td'):                
                        
                        T = tbody.find_all('tr')[i].find_all('td')[1].get_text()
                        TM = tbody.find_all('tr')[i].find_all('td')[2].get_text()
                        tm = tbody.find_all('tr')[i].find_all('td')[3].get_text()
                        H = tbody.find_all('tr')[i].find_all('td')[5].get_text()
                        P = tbody.find_all('tr')[i].find_all('td')[6].get_text()
                        VV = tbody.find_all('tr')[i].find_all('td')[7].get_text()
                        V = tbody.find_all('tr')[i].find_all('td')[8].get_text()
                        VM = tbody.find_all('tr')[i].find_all('td')[9].get_text()
                        csv_writer.writerow([T,TM,tm,H,P,VV,V,VM])
                    
                    else:
                        break

# AQI from mean generation
overall=[]
for year in range(2013,2019):
    temp_i=0  
    for rows in pd.read_csv('AQI\\aqi{}.csv'.format(year),chunksize=24):
        add_var=0
        avg=0.0
        data=[]
        df=pd.DataFrame(data=rows)
        for index,row in df.iterrows():
            data.append(row['PM2.5'])
        for i in data:
            if type(i) is float or type(i) is int:
                add_var=add_var+i
            elif type(i) is str:
                if i!='NoData' and i!='PwrFail' and i!='---' and i!='InVld':
                    temp=float(i)
                    add_var=add_var+temp
        avg=add_var/24
        temp_i=temp_i+1
        overall.append(avg)

df2 = pd.DataFrame(data=overall,columns=['PM2.5'])          
                    

df1 = pd.read_csv('indep.csv')

final = df1.join(df2)

final.dropna(inplace=True)


for col in final.columns:
    final[col] = np.where(final[col] == '-', 0, final[col])
    final[col] = pd.to_numeric(final[col], errors='coerce').fillna(0).astype(np.float64)
    
    if col != 'P':
        final[col] = np.where(final[col] == 0, final[col].mean(), final[col])
        

from sklearn.model_selection import train_test_split
X = final.drop(['PM2.5'],axis=1)
y = final['PM2.5']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # create an object
scaled_df = scaler.fit_transform(X)
X = pd.DataFrame(scaled_df,columns=X.columns)

from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X_train,y_train)
pred = lin.predict(X_test)



import pickle
# open a file, where you ant to store the data
file = open('linear_model.pkl', 'wb')

# dump information to that file
pickle.dump(lin, file)









