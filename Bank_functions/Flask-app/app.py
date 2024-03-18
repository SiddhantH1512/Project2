from flask import Flask, render_template, request
import sys
sys.path.append('/Users/siddhant/Project2/Bank_functions')
from utils import open_object
import pandas as pd

# Importing module 1 objects
model = open_object('Bank_functions/src/models/trained_models/final_model_module2.pkl')
scaler = open_object('/Users/siddhant/Project2/Bank_functions/src/models/transformer_objects/module2_scaler.pkl')

# Importing module 2 objects
pipeline = open_object('/Users/siddhant/Project2/Bank_functions/src/models/trained_models/final_model_module1.pkl')
transformer = open_object('/Users/siddhant/Project2/Bank_functions/src/models/transformer_objects/transformer_module1.pkl')

# Importing module 3 objects
transformer2 = open_object('/Users/siddhant/Project2/Bank_functions/src/models/transformer_objects/transformer_module3.pkl')
resampled_model = open_object('/Users/siddhant/Project2/Bank_functions/src/models/trained_models/final_model_resampled_module3.pkl')
imbalanced_model = open_object('/Users/siddhant/Project2/Bank_functions/src/models/trained_models/final_model_imbalanced_module3.pkl')

app = Flask(__name__, template_folder='/Users/siddhant/Project2/Bank_functions/Flask-app/template')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/loan_conversion', methods=['GET', 'POST'])
def loan_conversion():
    if request.method == 'POST':
        try:
            # Extract and convert form data, handle conversion to the appropriate data type
            age = int(request.form['Age'])
            income = float(request.form['Income'])
            family = int(request.form['Family'])
            ccavg = float(request.form['Credit Card usage (annual in thousand)'])
            education = int(request.form['Education'])
            cd_account = int(request.form['Certificate of Deposit account'])
            
            # Create a DataFrame for scaling
            df = pd.DataFrame(data=[[age, income, family, ccavg, education, cd_account]], 
                              columns=['Age', 'Income', 'Family', 'CCAvg', 'Education', 'CD Account'])
            
            # Scale data
            scaled_response = scaler.transform(df)
            
            # Make prediction
            prediction = model.predict(scaled_response)
            
            # Convert numerical prediction to a meaningful response
            prediction_text = 'likely' if prediction[0] == 1 else 'unlikely'
            
            # Render the same template but with additional context (prediction result)
            return render_template('loan_conversion.html', prediction=prediction_text)
        except Exception as e:
            # Render the template with an error message
            return render_template('loan_conversion.html', error=str(e))

    # If not POST method, just render the page without prediction
    return render_template('loan_conversion.html')




@app.route("/fraud_classification", methods=['GET', 'POST'])
def fraud_transaction():
    if request.method == 'POST':
        print("POST request received")
        try:
            # Extract and convert form data
            step = int(request.form['step'])
            type_trans = str(request.form['type'])
            amount = float(request.form['amount'])
            flagged = int(request.form['isFlaggedFraud'])
            balance_change = float(request.form['org_balance_change'])
            receiver_bal = float(request.form['dest_balance'])
            receiver_bal_change = float(request.form['dest_balance_change'])

            # Convert input data into a DataFrame
            try:
                df1 = pd.DataFrame([[step, type_trans, amount, flagged, balance_change, receiver_bal, receiver_bal_change]],
                                    columns=['step', 'type', 'amount', 'isFlaggedFraud', 'org_balance_change', 'dest_balance', 'dest_balance_change'])
                print("DataFrame created:", df1)
            except Exception as e:
                print(f"Error creating DataFrame: {e}")
                
            try:
                transformed_input = transformer.transform(df1)
                print('Input transformed')
            except Exception as e:
                print(f'Error during transformation: {e}')
                return render_template('fraud_classification.html', error=f"Transformation error: {e}")
                
            try:
                prediction2 = pipeline.predict(transformed_input)
                print(f"Raw Prediction Output: {prediction2}")
                prediction_info = 'Fraudulent transaction' if prediction2[0] == 1 else 'Normal transaction'
            except Exception as e:
                prediction_info = f"Error during prediction: {e}"


            # Render template with prediction info
            return render_template('fraud_classification.html', prediction2=prediction_info)
        except Exception as e:
            return render_template('fraud_classification.html', error=str(e))
    return render_template('fraud_classification.html')



@app.route('/customer_churn', methods=['GET', 'POST'])
def churn():
    if request.method == 'POST':
        print('Post request received')
        
        try:
            # Extracting input data from form
            credit_score = int(request.form['CreditScore'])
            region = str(request.form['Geography'])
            gender = str(request.form['Gender'])
            age = int(request.form['Age'])
            tenure = int(request.form['Tenure'])
            bank_balance = float(request.form['Balance'])
            no_of_products = int(request.form['NumOfProducts'])
            creditcard_possession = int(request.form['HasCrCard'])
            active_member = int(request.form['IsActiveMember'])
            salary = float(request.form['EstimatedSalary'])
            satisfaction_score = int(request.form['Satisfaction Score'])
            type_of_card = str(request.form['Card Type'])
            points_earned = str(request.form['Point Earned'])
            
            try:
                df2 = pd.DataFrame([[credit_score, region, gender, age, tenure, bank_balance, no_of_products, creditcard_possession, 
                                     active_member, salary, satisfaction_score, type_of_card, points_earned]], 
                                   columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                                            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Satisfaction Score', 'Card Type', 'Point Earned'])
                print('Dataframe created', df2)
            except Exception as e:
                print(f"Error creating DataFrame: {e}")
                
            try:
                transformed_input2 = transformer2.transform(df2)
                print('Transformed input column names:', transformed_input2.columns)

            except Exception as e:
                print(f'Error during transformation: {e}')
                return render_template('customer_churn.html', error=f"Transformation error: {e}")
            
            try:
                prediction3 = imbalanced_model.predict(transformed_input2)
                print(f"Raw Prediction Output: {prediction3}")
                prediction_info2 = 'likely churn' if prediction3[0] == 1 else 'not likely'
            except Exception as e:
                prediction_info2 = f"Error during prediction: {e}"
                
            # Render template with prediction info
            return render_template('customer_churn.html', prediction3=prediction_info2)
        except Exception as e:
            return render_template('customer_churn.html', error=str(e))
    return render_template('customer_churn.html')
    

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
