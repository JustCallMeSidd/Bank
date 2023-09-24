import numpy as np
import pandas as pd
import tensorflow as tf
import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('E:/ANN INTERN/Churn_Modelling.csv') 
X = dataset.iloc[:, 3:-1].values
Y = dataset.iloc[:, -1].values

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size=32, epochs=100)

app = tk.Tk()
app.title("Churn Prediction")


tk.Label(app, text="Geography:").grid(row=0, column=0)
geo_var = tk.StringVar()
geo_combobox = ttk.Combobox(app, textvariable=geo_var, values=["France", "Germany", "Spain"])
geo_combobox.grid(row=0, column=1)

tk.Label(app, text="Gender:").grid(row=1, column=0)
gender_var = tk.StringVar()
gender_combobox = ttk.Combobox(app, textvariable=gender_var, values=["Male", "Female"])
gender_combobox.grid(row=1, column=1)

tk.Label(app, text="Credit Score:").grid(row=2, column=0)
credit_score_var = tk.StringVar()
credit_score_entry = tk.Entry(app, textvariable=credit_score_var)
credit_score_entry.grid(row=2, column=1)

tk.Label(app, text="Age:").grid(row=3, column=0)
age_var = tk.StringVar()
age_entry = tk.Entry(app, textvariable=age_var)
age_entry.grid(row=3, column=1)

tk.Label(app, text="Tenure:").grid(row=4, column=0)
tenure_var = tk.StringVar()
tenure_entry = tk.Entry(app, textvariable=tenure_var)
tenure_entry.grid(row=4, column=1)

tk.Label(app, text="Balance:").grid(row=5, column=0)
balance_var = tk.StringVar()
balance_entry = tk.Entry(app, textvariable=balance_var)
balance_entry.grid(row=5, column=1)

tk.Label(app, text="Number of Products:").grid(row=6, column=0)
num_of_products_var = tk.StringVar()
num_of_products_entry = tk.Entry(app, textvariable=num_of_products_var)
num_of_products_entry.grid(row=6, column=1)

credit_card_var = tk.IntVar()
tk.Checkbutton(app, text="Credit Card", variable=credit_card_var).grid(row=7, column=0)

active_member_var = tk.IntVar()
tk.Checkbutton(app, text="Active Member", variable=active_member_var).grid(row=7, column=1)

tk.Label(app, text="Estimated Salary:").grid(row=8, column=0)
estimated_salary_var = tk.StringVar()
estimated_salary_entry = tk.Entry(app, textvariable=estimated_salary_var)
estimated_salary_entry.grid(row=8, column=1)

def predict_churn():
    gender_encoded = 1 if gender_var.get() == 'Male' else 0
    credit_score = float(credit_score_var.get())
    age = float(age_var.get())
    tenure = float(tenure_var.get())
    balance = float(balance_var.get())
    num_of_products = float(num_of_products_var.get())
    credit_card = float(credit_card_var.get())
    active_member = active_member_var.get()
    estimated_salary = float(estimated_salary_var.get())

    if geo_var.get() == 'France':
        input_data =[[1,0,0, gender_encoded, credit_score, age, tenure, balance, num_of_products,
                            credit_card, active_member, estimated_salary]]
    elif geo_var.get() == 'Germany':
        input_data =[[0,0,0, gender_encoded, credit_score, age, tenure, balance, num_of_products,
                            credit_card, active_member, estimated_salary]]
    else:
        input_data =[[0,0,1, gender_encoded, credit_score, age, tenure, balance, num_of_products,
                            credit_card, active_member, estimated_salary]]


    input_data = sc.transform(input_data)  


    churn_probability = ann.predict(input_data)

    if churn_probability > 0.5:
        result_label.config(text="We should not say goodbye to that customer.")
    else:
        result_label.config(text="We should say goodbye to that customer.")


predict_button = tk.Button(app, text="Predict Churn", command=predict_churn)
predict_button.grid(row=9, columnspan=2)

result_label = tk.Label(app, text="")
result_label.grid(row=10, columnspan=2)

app.mainloop()
