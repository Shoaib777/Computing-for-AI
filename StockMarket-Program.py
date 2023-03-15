from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as tk
import matplotlib.pyplot as pl
from sklearn import tree
import statsmodels.api as sm
import pandas as pd
import numpy as np
import urllib.parse
from pymongo import MongoClient
import json


def main():
    try:
        url = "cluster0.eaciurr.mongodb.net/?retryWrites=true&w=majority"
        Connection = MongoClient(f"mongodb+srv://Shoaib:{urllib.parse.quote('Khan@1235')}@{url}")
    except Exception:
        print("Connection to the Server is unsuccessful")
        return

        # Reading the CSV file and converting it into Json File
    df = pd.read_csv("Stock market data.csv")
    print(df.head(10))
    df.to_json("Stock market data.json")
    myTestingJSonData = open("Stock market data.json")
    jsonData = json.load(myTestingJSonData)

    # inserting the Json data to the mongoDB
    DB = Connection["Data"]
    Col = DB["Stock"]
    Col.insert_many([jsonData])
    print("Json Data Inserted")

    # Dropping the Created Collection
    print(f'The collection {Col} is dropped sucessfully')

    pl.hist((df['Interest_Rate'], df['Stock_Index_Price']))
    pl.title('Stock Index Price Vs Interest Rate', fontsize=14)
    pl.xlabel('Interest Rate', fontsize=14)
    pl.ylabel('Stock Index Price', fontsize=14)
    pl.grid(True)
    pl.show()

    pl.hist((df['Unemployment_Rate'], df['Stock_Index_Price']))

    pl.title('Stock Index Price Vs Unemployment Rate', fontsize=16)
    pl.xlabel('Unemployment Rate', fontsize=16)
    pl.ylabel('Stock Index Price', fontsize=16)
    pl.grid(True)
    pl.show()

    # here we have 2 variables for multiple regression.
    # If you just want to use one variable for simple linear regression,
    # then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
    X = df[['Interest_Rate', 'Unemployment_Rate']]
    Y = df['Stock_Index_Price']

    # with sklearn
    regr = tree.DecisionTreeRegressor()
    regr.fit(X, Y)


    # prediction with sklearn
    New_Interest_Rate = 2.75
    New_Unemployment_Rate = 5.3
    # print('Predicted Stock Index Price: \n', regr.predict(
    #     [[New_Interest_Rate, New_Unemployment_Rate]]))

    # with statsmodels
    X = sm.add_constant(X)  # adding a constant

    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X)
    print_model = model.summary()

    # here we have 2 input variables for multiple regression.
    # If you just want to use one variable for simple linear regression,
    # then use X = df['Interest_Rate'] for example.Alternatively,
    # you may add additional variables within the brackets
    X = df[['Interest_Rate', 'Unemployment_Rate']]
    Y = df['Stock_Index_Price']  # output variable (what we are trying to predict)

    # with sklearn
    regr = tree.DecisionTreeRegressor()
    regr.fit(X, Y)

    # with statsmodels
    X = sm.add_constant(X)  # adding a constant

    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X)

    # tkinter GUI
    root = tk.Tk()

    canvas1 = tk.Canvas(root, width=700, height=500)
    canvas1.pack()

    label1 = tk.Label(
        root, text='Your dataset has been loaded now you can predict the stock index price')
    canvas1.create_window(320, 200, window=label1)

    #
    button1 = tk.Button(root, text='Click Here to Predict the Stock Index Price',
                        command=root.destroy, bg='orange')  # button to call the 'values' command above
    canvas1.create_window(320, 250, window=button1)

    root.mainloop()

    # here we have 2 input variables for multiple regression.
    # If you just want to use one variable for simple linear regression,
    # then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
    X = df[['Interest_Rate', 'Unemployment_Rate']].astype(float)
    # output variable (what we are trying to predict)
    Y = df['Stock_Index_Price'].astype(float)

    # with sklearn
    regr = tree.DecisionTreeRegressor()
    regr.fit(X, Y)

    # print('Intercept: \n', regr.intercept_)
    # print('Coefficients: \n', regr.coef_)

    # tkinter GUI
    root = tk.Tk()

    canvas1 = tk.Canvas(root, width=500, height=300)
    canvas1.pack()
    root.title("GUI Interface")

    # with sklearn
    # Intercept_result = ('Intercept: ', regr.intercept_)
    label_Intercept = tk.Label(root,
                               # text=Intercept_result,
                               justify='center')
    canvas1.create_window(260, 220, window=label_Intercept)

    # with sklearn
    # Coefficients_result = ('Coefficients: ', regr.coef_)
    label_Coefficients = tk.Label(root,
                                  # text=Coefficients_result,
                                  justify='center')
    canvas1.create_window(260, 240, window=label_Coefficients)

    # New_Interest_Rate label and input box
    label1 = tk.Label(root, text='Type Interest Rate: ')
    canvas1.create_window(100, 100, window=label1)

    entry1 = tk.Entry(root)  # create 1st entry box
    canvas1.create_window(270, 100, window=entry1)

    # New_Unemployment_Rate label and input box
    label2 = tk.Label(root, text=' Type Unemployment Rate: ')
    canvas1.create_window(120, 120, window=label2)

    entry2 = tk.Entry(root)  # create 2nd entry box
    canvas1.create_window(270, 120, window=entry2)

    def values():
        global New_Interest_Rate  # our 1st input variable
        New_Interest_Rate = float(entry1.get())

        global New_Unemployment_Rate  # our 2nd input variable
        New_Unemployment_Rate = float(entry2.get())

        Prediction_result = ('Predicted Stock Index Price: ', regr.predict(
            [[New_Interest_Rate, New_Unemployment_Rate]]))
        label_Prediction = tk.Label(root, text=str(Prediction_result), bg='orange')
        canvas1.create_window(260, 280, window=label_Prediction)

    # button to call the 'values' command above
    button1 = tk.Button(root, text='Predict Stock Index Price',
                        command=values, bg='yellow')
    canvas1.create_window(270, 150, window=button1)

    # plot 1st scatter
    x = np.random.normal(df['Interest_Rate'], df['Stock_Index_Price'])
    figure3 = plt.Figure(figsize=(5, 4), dpi=100)
    ax3 = figure3.add_subplot(111)
    ax3.hist(x, rwidth=0.7, color="lightblue", ec="blue")
    hist3 = FigureCanvasTkAgg(figure3, root)
    hist3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
    ax3.legend()
    ax3.set_xlabel('Interest Rate')
    ax3.set_title('Interest Rate Vs. Stock Index Price')

    # plot 2nd scatter

    y = np.random.normal(df['Unemployment_Rate'], df['Stock_Index_Price'])
    figure4 = plt.Figure(figsize=(5, 4), dpi=100)
    ax4 = figure4.add_subplot(111)
    ax4.hist(y, rwidth=0.7, color="lightgreen", ec="green")
    hist4 = FigureCanvasTkAgg(figure4, root)
    hist4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
    ax4.legend()
    ax4.set_xlabel('Unemployment_Rate')
    ax4.set_title('Unemployment_Rate Vs. Stock Index Price')

    root.mainloop()


if __name__ == '__main__':
    main()
