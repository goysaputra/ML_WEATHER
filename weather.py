import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



def plot_temperature_distribution(data):
    plt.figure(figsize=(15, 4))
    sns.set_style("darkgrid", {"axes.facecolor": "0.9", 'grid.color': '.6', 'grid.linestyle': '-.'})
    temperature_dist = data.groupby("Date")["Temperature"].mean()

    temperature_dist.plot(kind='bar', rot=0)
    plt.xlabel("Temperature min", fontsize=14, color="r")
    plt.ylabel("Count", fontsize=14, color="r")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Temperature Distribution", fontsize=16, color="r")

    return plt.gcf()

def train_linear_model(data):
    X = data[['temp_max',]]
    y = data['wind']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test

def plot_weather_distribution(data):
    weather_data = pd.read_csv('seattle-weather.csv')  
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='temp_max', y='temp_min', data=data)
    plt.title('Weather Distribution')
    plt.xlabel('Temperature')
    plt.ylabel('Count')
    st.pyplot(plt.gcf())
    print(weather_data.head())
    print(weather_data.info())


def main():
    st.title("Weather Prediction App")

   
    weather_data = pd.read_csv('seattle-weather.csv')  

    plot_weather_distribution(weather_data)

    model, X_test, y_test = train_linear_model(weather_data)

    st.header("Temperature Prediction")
    temperature = st.slider("Select Temperature", min_value=int(weather_data['temp_min'].min()), 
                           max_value=int(weather_data['temp_max'].max()), step=1)

    prediction_input = pd.DataFrame({'temperature': [temperature]})
    prediction = model.predict(X_test)

    pred_str = np.array(prediction)
    pred_float = float(pred_str[0])

    st.write(f"Predicted Count: {pred_float:.2f}")

if __name__ == "__main__":
    main()
