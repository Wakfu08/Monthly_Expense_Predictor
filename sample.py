from flask import Flask, jsonify, make_response
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)


# Define the API endpoint
@app.route('/')
def predict_expenses():
    # Load data from CSV file
    df = pd.read_csv('sample.csv')

    # Convert Date column to datetime type
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

    # Extract month and category from Date and Category columns
    df['Month'] = df['Date'].dt.month
    df['Category'] = df['Category'].astype('category')

    # Group data by month and category and calculate total expense
    expense_df = df.groupby(['Month', 'Category'], as_index=False)['Expense'].sum()

    # Pivot table to create columns for each category and rows for each month
    expense_pivot = pd.pivot_table(expense_df, values='Expense', index='Month', columns='Category', fill_value=0)

    # Separate the last month's data for prediction
    last_month = expense_pivot.iloc[-1]
    training_data = expense_pivot.iloc[:-1]

    # Train a random forest regression model for each category
    models = {}
    for col in training_data.columns:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(training_data.drop(col, axis=1), training_data[col])
        models[col] = model

    # Use the trained models to predict the next month's expenses
    next_month = {}
    for col, model in models.items():
        next_month[col] = model.predict(last_month.drop(col).values.reshape(1, -1))[0]
    resp = make_response(jsonify(next_month))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
    # return jsonify(next_month)


if __name__ == '__main__':
    app.run(debug=True)
