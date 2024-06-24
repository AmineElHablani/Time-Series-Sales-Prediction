#libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score ,mean_squared_error
from tqdm import tqdm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

plt.rcParams['figure.figsize'] = (20, 7)



class Samples:
    #create samples (SALES)
    def segmentData(self,data,columns,segment,frequency=None):
        date_index = pd.date_range(data["Order Date"].min(),data["Order Date"].max())
        new_data = pd.pivot_table(data=data,index="Order Date",columns=columns,values=segment,aggfunc="sum")
        new_data = new_data.reindex(date_index,fill_value=0)
        if(frequency != None):
            new_data = new_data.resample(frequency).sum()

        #filling with 0 because there is 0 purshases
        new_data = new_data.fillna(0)
        return new_data 

    
    
    #create samples (SALES)
    def salesData(self,data,columns,frequency=None):
        date_index = pd.date_range(data["Order Date"].min(),data["Order Date"].max())
        new_data = pd.pivot_table(data=data,index="Order Date",columns=columns,values="Sales",aggfunc="sum")
        new_data = new_data.reindex(date_index,fill_value=0)
        if(frequency != None):
            new_data = new_data.resample(frequency).sum()

        #filling with 0 because there is 0 purshases
        new_data = new_data.fillna(0)
        return new_data 
    
    
    #create samples (Profit) 
    def profitData(self,data,columns,frequency=None):
        date_index = pd.date_range(data["Order Date"].min(),data["Order Date"].max())
        
        new_data = pd.pivot_table(data=data,index="Order Date",columns=columns,values="Profit",aggfunc="sum")
        new_data = new_data.reindex(date_index,fill_value=0)
        if(frequency != None):
            new_data = new_data.resample(frequency).sum()
        #filling with 0 because there is 0 purshases
        new_data = new_data.fillna(0)
        return new_data
    def generalData(self,data,frequency):
        general_sales_data = data.groupby("Order Date")["Sales"].sum().reset_index()
        general_sales_data.set_index("Order Date",inplace=True)
        date_index = pd.date_range(data["Order Date"].min(),data["Order Date"].max())
        general_sales_data = general_sales_data.reindex(date_index,fill_value=0)
        if(frequency != None):
            general_sales_data = general_sales_data.resample(frequency).sum()
        return general_sales_data
            
    

class visualization:
    
    def lineplot(self,data,frequency,value=""):
        #we use pseudo addive methode because we got many 0 values
        #decompose = seasonal_decompose(data[data.columns[i]], model="Pseudo-additive")
        number_columns = len(data.columns)
        if(number_columns == 1):
            figsize=(20,7)
        else:
            figsize=(20,15)
        fig,axis = plt.subplots(len(data.columns),1,figsize=figsize)
        
        if(number_columns == 1):
            #line plot
            axis.plot(data[data.columns[0]])
            axis.set_title(f"{frequency.title()} {value} - {data.columns[0]}",fontsize=30)
        else:
            for i in range(len(data.columns)):
                #line plot
                axis[i].plot(data[data.columns[i]])
                axis[i].set_title(f"{frequency.title()} {value} - {data.columns[i]}",fontsize=30)        
        plt.tight_layout()
        
    def trend_plot(self, data,frequency,value = ""):
        #we use pseudo addive methode because we got many 0 values
        number_columns = len(data.columns)
        if(number_columns == 1):
            figsize=(20,7)
        else:
            figsize=(20,15)
        fig,axis = plt.subplots(len(data.columns),1,figsize=figsize)
        
        if(number_columns == 1):
                decompose = seasonal_decompose(data[data.columns[0]], model="Pseudo-additive")
                #plot
                axis.plot(decompose.trend)
                axis.set_title(f"{frequency.title()} {value} - Trend - {data.columns[0]}",fontsize=30)
        else:
            for i in range(len(data.columns)):
                decompose = seasonal_decompose(data[data.columns[i]], model="Pseudo-additive")
                #plot
                axis[i].plot(decompose.trend)
                axis[i].set_title(f"{frequency.title()} {value} - Trend - {data.columns[i]}",fontsize=30)

        plt.tight_layout()
    def seasonality_plot(self, data,frequency,value=""):
        #we use pseudo addive methode because we got many 0 values
        number_columns = len(data.columns)
        if(number_columns == 1):
            figsize=(20,7)
        else:
            figsize=(20,15)
        fig,axis = plt.subplots(len(data.columns),1,figsize=figsize)
        
        if(number_columns == 1):
                decompose = seasonal_decompose(data[data.columns[0]], model="Pseudo-additive")
                #plot
                axis.plot(decompose.seasonal)
                axis.set_title(f"{frequency.title()} {value} - Seasonality - {data.columns[0]}",fontsize=30)
                axis.set_xticks(data.index.date, data.index.date, rotation = 90)
        else:
            for i in range(len(data.columns)):
                decompose = seasonal_decompose(data[data.columns[i]], model="Pseudo-additive")
                #plot
                axis[i].plot(decompose.seasonal)
                axis[i].set_title(f"{frequency.title()} {value} - Seasonality - {data.columns[i]}",fontsize=30)
                axis[i].set_xticks(data.index.date, data.index.date, rotation = 90)
        plt.tight_layout()
        
    def residual_plot(self, data,frequency,value=""):
        #we use pseudo addive methode because we got many 0 values
        number_columns = len(data.columns)
        if(number_columns == 1):
            figsize=(20,7)
        else:
            figsize=(20,15)
        fig,axis = plt.subplots(len(data.columns),1,figsize=figsize)
        
        if(number_columns == 1):
            decompose = seasonal_decompose(data[data.columns[0]], model="Pseudo-additive")
            #plot
            axis.plot(decompose.resid)
            axis.set_title(f"{frequency.title()} {value} - residual - {data.columns[0]}",fontsize=30)
        else:
            for i in range(len(data.columns)):
                decompose = seasonal_decompose(data[data.columns[i]], model="Pseudo-additive")
                #plot
                axis[i].plot(decompose.resid)
                axis[i].set_title(f"{frequency.title()} {value} - residual - {data.columns[i]}",fontsize=30)

        plt.tight_layout()
    
    def monthly_plot(self, data,value=""):
        #we use pseudo addive methode because we got many 0 values
        number_columns = len(data.columns)
        if(number_columns == 1):
            figsize=(15,7)
        else:
            figsize=(15,15)
        fig,axis = plt.subplots(len(data.columns),1,figsize=figsize)
        
        if(number_columns == 1):
            month_plot(data[data.columns[0]], ax = axis)
            axis.set_title(f"Monthly {value} - {data.columns[0]} {value}",fontsize=30)
        else:
            for i in range(len(data.columns)):
                #plot
                month_plot(data[data.columns[i]], ax = axis[i])
                axis[i].set_title(f"Monthly {value} - {data.columns[i]} {value}",fontsize=30)
        plt.tight_layout()
        
    def quarterly_plot(self, data,value=""):
        #we use pseudo addive methode because we got many 0 values
        number_columns = len(data.columns)
        if(number_columns == 1):
            figsize=(15,7)
        else:
            figsize=(15,15)
        fig,axis = plt.subplots(len(data.columns),1,figsize=figsize)
        
        if(number_columns == 1):
            quarter_plot(data[data.columns[0]], ax = axis[0])
            axis.set_title(f"Quarterly {value} - {data.columns[0]}",fontsize=30)
        else:
            for i in range(len(data.columns)):
            #plot
                quarter_plot(data[data.columns[i]], ax = axis[i])
                axis[i].set_title(f"Quarterly {value} - {data.columns[i]}",fontsize=30)
        plt.tight_layout()
            
    def acf_plot(self, data ,frequency,value=""):
        #we use pseudo addive methode because we got many 0 values
        #fig,axis = plt.subplots(len(data.columns),1,figsize=(15,15))
        for i in range(len(data.columns)):
            #plot
            plot_acf(data[data.columns[i]], title=f'Autocorrelation in {data.columns[i]} {frequency} {value} Data')
        plt.tight_layout()
        
    def pacf_plot(self, data,frequency ,value=""):
        #we use pseudo addive methode because we got many 0 values
        #fig,axis = plt.subplots(len(data.columns),1,figsize=(15,15))
        for i in range(len(data.columns)):
            #plot
            plot_pacf(data[data.columns[i]], title=f'Partial autocorrelation in {data.columns[i]} {frequency} {value} Data')
        plt.tight_layout()
            

class Train:
    def split_data(self,data,percentage):
        train_size = round(data.shape[0] * percentage)
        train_end= data.index[train_size -1]
        test_start= data.index[train_size]
        X_train = data.loc[:train_end]
        X_test = data.loc[test_start:]
        return X_train, X_test 
    
    def ACF(sef,data):
        for i in range(len(data.comuns)):
            acf_plot = plot_acf(data[data.columns[i]], title=f'Autocorrelation in {data.columns[i]} Monthly Sales Data')

    def PACF(sef,data):
        for i in range(len(data.comuns)):
            acf_plot = plot_pacf(data[data.columns[i]], title=f'Autocorrelation in {data.columns[i]} Monthly Sales Data')
    
    def Evaluation(self,y_train,y_test):
        evaluation = {'Metrics':["MAE","MAPE","R2_SCORE"],
                      "Values":[mean_absolute_error(y_train,y_test),mean_absolute_percentage_error(y_train,y_test),r2_score(y_train,y_test)]}
        return pd.DataFrame(evaluation)
    
    def adf_test(self,data):
        result = adfuller(data)
        output = {
            "Statistical test": [result[0]],
            "P-values":[result[1]],
            "Lags used":[result[2]],
            "Number of observations":[result[3]]
        }
        return pd.DataFrame(output)
        
    def cumsum(self,liste):
        for i in range(1,len(liste)):
            liste[i] += liste[i-1]
        return liste
    def reverse_diff(self,y_train_data,y_diff_data,d,subset):
        #get copy 
        y_data = y_train_data.copy()
        y_diff = y_diff_data.copy()
        #save y_last (that was removed while differenciating)
        indexes=[] #datetime
        y_last=[]
        for i in range(d):
            indexes.append(y_data.index[0])
            y_last.append(y_data[0])
            y_data = y_data.diff().dropna()
        #reverse y_last to simplify code 
        y_last.reverse()
        indexes.reverse()

        #concatenate the differenciated data_train + pred/test
        if(subset=="test"):
            y_data = pd.concat([y_data,y_diff],axis=0)
        else:
            y_data = y_diff.copy()
            

        #reverse 
        for i in range(d):
            y_data.at[indexes[0].to_pydatetime()] = y_last[i] 
            #print(y_data)
            y_data.sort_index(inplace=True)
            y_data = self.cumsum(y_data)
        #if(subset == "test"):    
        y_diff=y_data[-(len(y_diff)):]
        #else:
        #    y_diff = y_data
            
        return y_diff 
    
    def findSeasonality(self,acf_list,pacf_list):
        results=[]
        for ma in acf_list:
            if ma in pacf_list:
                results.append(ma)
        return results
        
    def rolling_forecast_Arima(self,data,percentage_test,order):
        #get size of train and test samples
        train_size = round(len(data) * percentage_test)
        test_size= len(data) - train_size
        #get train and test initial samples 
        test_dates = data[train_size:].index
        
        #train and predict
        y_pred=[]
        for train_end in test_dates:
            train = data[:train_end]
            model = ARIMA(train,order=order)
            model = model.fit()
            pred = model.forecast(1)
            y_pred.append(pred[0])
        return y_pred
    
    def listToPandas(self,target_type,liste):
        target_copy = target_type.copy()
        target_type["liste"] = liste 
        return target_type["liste"]
    
    def train_arima(self,data,split_percentage):
        #result 
        evaluation = {
            'Model':[],
            'Segment':[],
            
            'ACF':[],
            'd':[],
            'PACF':[],
            'Order':[],
            
            'MAE':[],
            'MAPE':[],
            'RMSE':[],
            'R2_Score':[],
            #'AIC':[]

            'MAE_train':[], 
            'MAPE_train':[],
            'R2_Score_train':[],
            'RMSE_train':[],
            
            #'MAE_forecast':[],
            #'MAPE_forecast':[],
            #'R2_Score_forecast':[],
            #'RMSE_forecast':[],
            
            'Y_test':[],
            'Y_prediction':[],
            "Y_prediction_train":[], 
            #'Y_prediction_rolling':[],
            #'Y_prediction_forecast':[],
        }
 
        #get columns => iterate over the columns (select the model's segment)
        for column in tqdm(data.columns): 
            temp_data = data.copy() 
            #save original train and test data : 
            data_train_origin , data_test_origin = self.split_data(temp_data,split_percentage)      

            
            #save train 
            #y_train= data_train[column]
            #check stationarity 
            result = adfuller(temp_data[column])
            print('ADF Statistic:', result[0])
            print('p-value:', result[1])
            d=0
            while(result[1] > 0.05):
                temp_data = temp_data.diff().dropna()
                #data_test = data_test.diff().dropna()
                result = adfuller(temp_data[column])
                d += 1 
            
            #split data
            data_train, data_test = self.split_data(temp_data,split_percentage)      

                
                
            
            #calculate ACF & PACF (choosed a threshhold = 25% and confidance interval = 95%)( _ = confidance interval)
            threshhold= 0.25
            acf_segment , _ = acf(temp_data[column],alpha=0.05)
            acf_segment = [lag for lag,value in enumerate(acf_segment) if abs(value)> threshhold]
            pacf_segment , _ = pacf(temp_data[column],alpha=0.05)

            pacf_segment = [lag for lag,value in enumerate(pacf_segment) if abs(value)> threshhold]

            
            if len(acf_segment) == 0 :
                print("no paramaters")
                #acf_segment=[0]
            if len(acf_segment) == 0 :
                #pacf_segment = [0]
                print("no paramaters")
                
            if (len(acf_segment) == 0) and (len(pacf_segment) == 0):
                print("no")
                continue
            print("acf",acf_segment)
            print("pacf",pacf_segment)
            
            #create ARIMA model for each segment(with all paramaters) (acf => ma , pacf => ar)
            for ar in pacf_segment:
                for ma in acf_segment:
                    try:
                        print(f"(ar,d,ma) : {ar,d,ma}")
                        #train the model
                        arima_model = ARIMA(data_train[column],order=(ar,d,ma))
                        #arima_model = arima_model.fit()
                        arima_model = arima_model.fit()
                        #y_pred = arima_model.predict(start=data_test.index[0],end=data_test.index[-1])
                        #y_pred = arima_model.forecast(steps=len(data_test[column]))
                        #prediction : rolling forecast origin (dynamic)
                        #y_pred_rolling = arima_model.predict(start=data_test.index[0],end=data_test.index[-1],dynamic=True)
                        #y_pred_rolling = self.rolling_forecast_Arima(temp_data[column],split_percentage,(ar,d,ma))
                        #forecast 
                        y_pred_forecast = arima_model.forecast(len(data_test[column]))

                        
                        #forecast on training (test valuate how the model behaves to known-training data)
                        #y_pred_train = arima_model.predict(start=data_train[column].index[0],end=data_train[column].index[-1])
                        #y_pred_train = arima_model.predict(start=data_train[column].index[0],end=data[column].index[-1])
                        y_pred_train = arima_model.predict()
                        
                        #temporary solution for nan predictions . until i find the main problem 
                        y_pred_forecast = pd.Series([0 if np.isnan(pred) else pred for pred in y_pred_forecast])
                        y_pred_forecast.index = data_test[column].index
                        
                        y_pred_train = pd.Series([0 if np.isnan(pred) else pred for pred in y_pred_train])
                        y_pred_train.index = data_train[column].index
                        

                        
                        #evaluation 
                        #normal prediction
                        #mae = mean_absolute_error(data_test[column],y_pred)
                        #mape = mean_absolute_percentage_error(data_test[column],y_pred)
                        #r2score = r2_score(data_test[column],y_pred)
                        #rmse = mean_squared_error(data_test[column],y_pred,squared=False)
                        order = (ar,d,ma)
                        
                        #rolling forecast orgin(dynamic)
                        #mae_rolling = mean_absolute_error(data_test[column],y_pred_rolling)
                        #mape_rolling = mean_absolute_percentage_error(data_test[column],y_pred_rolling)
                        #r2score_rolling = r2_score(data_test[column],y_pred_rolling)
                        #rmse_rolling = mean_squared_error(data_test[column],y_pred_rolling,squared=False)
                        
                        #evaluate training fit =========================
                        mae_train = mean_absolute_error(data_train[column],y_pred_train)
                        mape_train = mean_absolute_percentage_error(data_train[column],y_pred_train)
                        r2score_train = r2_score(data_train[column],y_pred_train)
                        rmse_train = mean_squared_error(data_train[column],y_pred_train,squared=False)
                        
                        #rolling forecast orgin(dynamic)
                        mae_forecast = mean_absolute_error(data_test[column],y_pred_forecast)
                        mape_forecast = mean_absolute_percentage_error(data_test[column],y_pred_forecast)
                        r2score_forecast = r2_score(data_test[column],y_pred_forecast)
                        rmse_forecast = mean_squared_error(data_test[column],y_pred_forecast,squared=False)

                        #add row . (save model infos)
                        evaluation['Model'].append(arima_model)
                        evaluation['Segment'].append(column)
                        
                        evaluation['ACF'].append(ma)
                        evaluation['d'].append(d)
                        evaluation['PACF'].append(ar)
                        evaluation['Order'].append(order)
                        #evaluation['AIC'].append(arima_model.aic)
                        
                        evaluation['MAE'].append(mae_forecast)
                        evaluation['MAPE'].append(mape_forecast)
                        evaluation['R2_Score'].append(r2score_forecast)
                        evaluation['RMSE'].append(rmse_forecast)
                        
                        
                        evaluation['MAE_train'].append(mae_train)
                        evaluation['MAPE_train'].append(mape_train)
                        evaluation['R2_Score_train'].append(r2score_train)
                        evaluation['RMSE_train'].append(rmse_train)
                        
                        
                        #evaluation['MAE_forecast'].append(mae_forecast)
                        #evaluation['MAPE_forecast'].append(mape_forecast)
                        #evaluation['R2_Score_forecast'].append(r2score_forecast)
                        
                        #transform list (y_pred) to pd  format
                        #y_pred_rolling = self.listToPandas(data_test,y_pred_rolling)
                        
                        #reverse diff
                        #simplify the code later (only if . without else) 
                        if(d != 0):
                            #y_pred_reversed = self.reverse_diff(data_train_origin[column],y_pred,d)
                            y_test_reversed = self.reverse_diff(data_train_origin[column],data_test[column],d,"test")
                            #y_pred_rolling_reversed = self.reverse_diff(data_train_origin[column],y_pred_rolling,d)
                            y_pred_forecast_reversed = self.reverse_diff(data_train_origin[column],y_pred_forecast,d,"test")
                            y_pred_train_reversed = self.reverse_diff(data_train_origin[column],y_pred_train,d,"train")
                            evaluation['Y_test'].append((y_test_reversed))
                            evaluation['Y_prediction'].append(y_pred_forecast_reversed)
                            evaluation['Y_prediction_train'].append(y_pred_train_reversed)
                            #evaluation['Y_prediction_rolling'].append(y_pred_rolling_reversed)                             
                            #['Y_prediction_forecast'].append(y_pred_forecast)                             
                        #save
                        else:
                            evaluation['Y_test'].append(data_test[column])
                            evaluation['Y_prediction'].append(y_pred_forecast)
                            evaluation['Y_prediction_train'].append(y_pred_train)
                            #evaluation['Y_prediction_rolling'].append(y_pred_rolling)
                            #evaluation['Y_prediction_forecast'].append(y_pred_forecast)                             
                            
                        
                        #evaluation['Y_test'].append(data_test[column])
                        #evaluation['Y_prediction'].append(y_pred)
                        #evaluation['Y_prediction_rolling'].append(y_pred_rolling) 
                    except  :
                        continue
                    #""" x """
        
        return pd.DataFrame(evaluation) 
    def initialize_sarima_dict(self):
        result = {
            'Model':[],
            'Segment':[],
            'ACF':[],
            'd':[],
            'PACF':[],
            'Order':[],
            "P":[],
            "D":[],
            "Q":[],
            "S":[],
            'MAE':[],
            'MAPE':[],
            'RMSE':[],
            'R2_Score':[],

            'MAE_train':[],
            'MAPE_train':[],
            'R2_Score_train':[],
            'RMSE_train':[],

            'Y_test':[],
            'Y_prediction':[],
            "Y_prediction_train":[], 
        }
        return result
    def result_sarima(self,data_train,data_test,data_train_origin,column_name,evaluation,p,d,q,upperP,D,upperQ,S):
        sarima_model = ARIMA(data_train,  
                                order = (p,d,q),
                                seasonal_order = (upperP,D,upperQ,S)
                                ).fit()
        #make forecasts 
        y_pred_train = sarima_model.predict()
        y_pred_test = sarima_model.forecast(len(data_test))

        #evaluate
        order = (p,d,q)
        
        #get train_subset(eliminet first S months)
        data_train = data_train[S:]
        y_pred_train = y_pred_train[S:]
        
        #temporary solution for nan predictions . until i find the main problem 
        indexes_train = y_pred_train.index
        indexes_test = y_pred_test.index
        
        y_pred_test = pd.Series([0 if np.isnan(pred) else pred for pred in y_pred_test])
        y_pred_test.index = indexes_test
        y_pred_train = pd.Series([0 if np.isnan(pred) else pred for pred in y_pred_train])
        y_pred_train.index = indexes_train
        #evaluate training fit 
        mae_train = mean_absolute_error(data_train,y_pred_train)
        mape_train = mean_absolute_percentage_error(data_train,y_pred_train)
        r2score_train = r2_score(data_train,y_pred_train)
        rmse_train = mean_squared_error(data_train,y_pred_train,squared=False)

        #rolling forecast orgin(dynamic)
        mae_forecast = mean_absolute_error(data_test,y_pred_test)
        mape_forecast = mean_absolute_percentage_error(data_test,y_pred_test)
        r2score_forecast = r2_score(data_test,y_pred_test)
        rmse_forecast = mean_squared_error(data_test,y_pred_test,squared=False)

        #add row . (save model infos)
        evaluation['Model'].append(sarima_model)
        evaluation['Segment'].append(column_name)

        evaluation['ACF'].append(q)
        evaluation['d'].append(d)
        evaluation['PACF'].append(p)
        evaluation['Order'].append(order)
        evaluation['P'].append(upperP)
        evaluation['Q'].append(upperQ)
        evaluation['D'].append(D)
        evaluation['S'].append(S)

        evaluation['MAE'].append(mae_forecast)
        evaluation['MAPE'].append(mape_forecast)
        evaluation['R2_Score'].append(r2score_forecast)
        evaluation['RMSE'].append(rmse_forecast)


        evaluation['MAE_train'].append(mae_train)
        evaluation['MAPE_train'].append(mape_train)
        evaluation['R2_Score_train'].append(r2score_train)
        evaluation['RMSE_train'].append(rmse_train)
        

        
        if(d != 0):
            data_test_reversed = self.reverse_diff(data_train_origin[column_name],data_test,d,"test")
            y_pred_test_reversed = self.reverse_diff(data_train_origin[column_name],y_pred_test,d,"test")
            y_pred_train_reversed = self.reverse_diff(data_train_origin[column_name],y_pred_train,d,"train")
            evaluation['Y_test'].append((data_test_reversed))
            evaluation['Y_prediction'].append(y_pred_test_reversed)
            evaluation['Y_prediction_train'].append(y_pred_train_reversed)

        #save
        else:
            evaluation['Y_test'].append(data_test)
            evaluation['Y_prediction'].append(y_pred_test)
            evaluation['Y_prediction_train'].append(y_pred_train)

        
        return pd.DataFrame(evaluation)
    
    def train_sarima(self,data,split_percentage):
        #result 
        evaluation = self.initialize_sarima_dict()
        evaluation = pd.DataFrame(evaluation)
        #get columns => iterate over the columns (select the model's segment)
        for column in tqdm(data.columns): 
            temp_data = data.copy() 
            #save original train and test data : 
            data_train_origin , data_test_origin = self.split_data(temp_data,split_percentage)      


            #save train 
            #y_train= data_train[column]
            #check stationarity 
            result = adfuller(temp_data[column])
            print('ADF Statistic:', result[0])
            print('p-value:', result[1])
            d=0
            while(result[1] > 0.05):
                temp_data = temp_data.diff().dropna()
                #data_test = data_test.diff().dropna()
                result = adfuller(temp_data[column])
                d += 1 

            #split data
            data_train, data_test = self.split_data(temp_data,split_percentage)      




            #calculate ACF & PACF (choosed a threshhold = 25% and confidance interval = 95%)( _ = confidance interval)
            threshhold= 0.25
            acf_segment , _ = acf(temp_data[column],alpha=0.05)
            acf_segment = [lag for lag,value in enumerate(acf_segment) if abs(value)> threshhold]
            pacf_segment , _ = pacf(temp_data[column],alpha=0.05)
            pacf_segment = [lag for lag,value in enumerate(pacf_segment) if abs(value)> threshhold]
            
            #solving error , temporary
            #pacf_segment = [value for value in pacf_segment if value > 0]
            #acf_segment = [value for value in acf_segment if value > 0]
            
            if len(acf_segment) == 0 :
               # acf_segment=[0]
               print("no paramaters")
            if len(acf_segment) == 0 :
               print("no paramaters")
                
                #pacf_segment = [0]
            if (len(acf_segment) == 0) and (len(pacf_segment) == 0):
                print("no")
                continue
            print("acf",acf_segment)
            print("pacf",pacf_segment)
            S_list = [s for s in pacf_segment if s in acf_segment]
            S_list = [s for s in S_list if s > 1]
            if 12 not in S_list :
                S_list.append(12)
            
            #avoid error of ar or ma > S 
            for p in pacf_segment :
                if p >= min(S_list):
                    if(1 not in pacf_segment):
                        pacf_segment.append(1)
                    if(min(S_list) - 1) not in pacf_segment:
                        pacf_segment.append(min(S_list) - 1)
                        
            for q in acf_segment :
                if q >= min(S_list):
                    if(1 not in acf_segment):
                        acf_segment.append(1)
                    if(min(S_list) - 1) not in acf_segment:
                        acf_segment.append(min(S_list) - 1)
            upperP_list= [0,1,2]
            Q_list =[0,1,2]

            
            #create ARIMA model for each segment(with all paramaters) (acf => ma , pacf => ar)
            for S in S_list:
                for ar in pacf_segment:
                    for ma in acf_segment:
                        #if ((ar == 0) and (ma == 0)):
                        if ((ma >= S) or (ar >= S)):
                            continue
                        #    continue
                            #if(S > max([ar,ma])):
                        for D in [1]:
                            #S = ar
                            #D = 1
                            for upperP in upperP_list:
                                for upperQ in Q_list:
                                    try:
                                        #contraint Q + P <=2
                                        if ((upperP != 0) or (upperQ != 0)):
                                            print(f"(ar,d,ma)(P,D,Q,S) : ({ar,d,ma})({upperP,D,upperQ,S})")
                                            empty_dict = self.initialize_sarima_dict()
                                            #fit sarima model
                                            row = self.result_sarima(data_train[column],data_test[column],data_train_origin,column,empty_dict,ar,d,ma,upperP,D,upperQ,S)
                                            evaluation = pd.concat([evaluation,row],axis=0)


                                    except  :
                                        print("failed")
                                        continue
                    #""" x """

        return evaluation           
                    
                    
                    
                    
                    