import pandas as pd
import numpy as np
from scipy.sparse.construct import random
#import seaborn as sns
#import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn import metrics

st.title('**ESP Rate Prediction**')
st.markdown('''
This program is able to predict **flow rate** from ESP well based on several input parameters.
* First, you need to either input your own dataset or use sample dataset
* Then, you need to select the model configuration
* Last, you need to input the value for each parameters

''')

st.sidebar.title('**Input Data Section**')
input = st.sidebar.selectbox('Choose Dataset: ',['Your Dataset','Sample Dataset'])

if input == 'Your Dataset':
    upload = st.sidebar.file_uploader("Drop your dataset (.xlsx): ")
    if upload is not None:
        df = pd.read_csv(upload)
    else:
        df = pd.read_excel('input combined.xlsx',index_col="date",parse_dates=["date"])

else:
    df = pd.read_excel('input combined.xlsx',index_col="date",parse_dates=["date"])

st.sidebar.write("*Note: required column in dataset are 'date' and 'rate")

st.sidebar.subheader('Model Input Section')
model_type = st.sidebar.selectbox('Regression Model Method', ['Linear Regression','Decision Tree Regressor','Random Forest Regressor'])
test_ratio = float(st.sidebar.slider('Test Data Ratio (%) ', min_value=0, max_value=100,value=33))



df = df.dropna()
X = df.drop('rate',axis=1)
y = df.rate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio/100, random_state=123)

#Linear Regression Method 

if model_type == 'Linear Regression':
    model = LinearRegression()
elif model_type == 'Decision Tree Regressor':
    model = DecisionTreeRegressor(random_state=123)
else:
    model = RandomForestRegressor(random_state=123)

model.fit(X_train,y_train)
predictions = model.predict(X_test)

#figg, ax = plt.subplots(figsize=(10, 10))
#plot_tree(model, fontsize=10)

#st.write(figg)




z = pd.DataFrame(columns = ['MAE','MSE','RMSE','R2'], 
            data=[[metrics.mean_absolute_error(y_test, predictions),
            metrics.mean_squared_error(y_test, predictions),
            np.sqrt(metrics.mean_squared_error(y_test, predictions)),
            metrics.r2_score(y_test,predictions)]])
            
input_predictions = []

st.sidebar.subheader('Prediction Input')
for i in X.columns:
    x = st.sidebar.number_input("Input {}".format(i),min_value=0)
    input_predictions.append(x)

#st.write(input_predictions)

output = float(model.predict([input_predictions]))


st.info('**Predicted Output** : {} '.format(output))
#st.write('Predicted flow rate: ', output)

input_predictions.append(output)

#st.write(input_predictions)

new = df.tail(1).reset_index().iloc[:,1:]
new.loc[1] = input_predictions

#st.write(new)

st.info('''**Model Evaluation Metrics** 


Mean Absolute Error (MAE): {}

Mean Squared Error (MSE): {}

Root Mean Squared Error (RMSE): {}

R Squared (R2): {}

'''.format(metrics.mean_absolute_error(y_test, predictions)
            ,metrics.mean_squared_error(y_test, predictions)
            ,np.sqrt(metrics.mean_squared_error(y_test, predictions))
            , metrics.r2_score(y_test,predictions)))
#st.write('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, predictions))
#st.write('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, predictions))
#st.write('Mean Squared Error (MSE):', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#st.write('R Squared (R2):', metrics.r2_score(y_test,predictions))

#st.write(pd.DataFrame(   columns = ['Mean Absolute Error','Mean Squared Error','Root Mean Squared Error','R Squared'], 
                    #data = [[metrics.mean_absolute_error(y_test, predictions)
                           # ,metrics.mean_squared_error(y_test, predictions)
                           # ,np.sqrt(metrics.mean_squared_error(y_test, predictions))
                           # ,metrics.r2_score(y_test,predictions)]]))





#Plotting Function
#st.write('**Regression Plot**')
chart = px.scatter(x=y_test,y=predictions,width=600,height=450)
chart.add_trace(go.Scatter(x=y_test,y=y_test,
                    mode='lines',
                    name='Best Fit Line'))

chart.update_layout(title={'text':'Regression Plot',
                            'xanchor' : 'left',
                            'yanchor' :'top',
                            'x' : 0},
                   xaxis_title='Test Data',
                   yaxis_title='Predicted Data')
st.plotly_chart(chart)

check_data = st.checkbox('Display Input Dataset')

if check_data:
    st.write(df)







