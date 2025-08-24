# LendingClub Loan Default Prediction (Deep Learning with Keras)

## 📌 Overview  
This project uses a deep learning model built with the **Keras API** to predict whether a borrower will fully repay their loan or default (charged-off). Using historical LendingClub loan data from Kaggle, the model learns from financial and personal features of borrowers to assess credit risk.  

LendingClub is the world’s largest peer-to-peer lending platform, and this dataset provides rich real-world loan information such as income, loan amount, interest rates, employment length, credit history, and repayment status.  

---

## 🎯 Goal  
The objective is to **predict loan repayment status (`Fully Paid` vs `Charged Off`)**.  
This has direct business impact: helping lenders reduce risk by identifying potentially defaulting borrowers before approving loans.  

---

## 📂 Dataset  
Dataset: [LendingClub Loan Data (Kaggle)](https://www.kaggle.com/wordsforthewise/lending-club)  

**Key Features**:  
- `loan_amnt`: Loan amount requested by the borrower  
- `term`: Loan term (36 or 60 months)  
- `int_rate`: Loan interest rate  
- `installment`: Monthly payment by borrower  
- `grade`, `sub_grade`: LendingClub loan risk categories  
- `emp_length`: Borrower’s employment length  
- `annual_inc`: Annual income  
- `home_ownership`: Rent/Own/Mortgage  
- `dti`: Debt-to-income ratio  
- `mort_acc`: Number of mortgage accounts  
- `pub_rec_bankruptcies`: Bankruptcies in credit history  

**Target**:  
- `loan_status` → Converted into `loan_repaid` (1 = Fully Paid, 0 = Charged Off)  

---

## 🔎 Exploratory Data Analysis (EDA)  
- Countplots of loan status and loan grades  
- Histograms of loan amount distributions  
- Heatmaps of correlations between financial features  
- Boxplots comparing loan amount vs repayment status  
- Analysis of subgrades, employment length, and default risk  

Key Insights:  
- Strong correlation between `installment` and `loan_amnt` (duplicate info).  
- Lower grades (F, G) are highly correlated with defaults.  
- Employment length has little effect on default probability.  
- Some features like `emp_title`, `title`, and `issue_d` were dropped as redundant.  

---

## ⚙️ Data Preprocessing  
- Dropped high-cardinality or redundant features (`emp_title`, `title`, etc.)  
- Handled missing values (e.g., filled `mort_acc` intelligently based on correlations)  
- Converted categorical features into dummy variables  
- Normalized continuous variables using **MinMaxScaler**  
- Train-test split for supervised learning  

---

## 🧠 Model Architecture (Keras Sequential API)  
```python
model = Sequential()
model.add(Dense(78, activation='relu'))  
model.add(Dropout(0.2))  
model.add(Dense(39, activation='relu'))  
model.add(Dropout(0.2))  
model.add(Dense(19, activation='relu'))  
model.add(Dense(1, activation='sigmoid'))  # Binary classification
```
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Metrics: Accuracy
- Trained for multiple epochs with validation data

## 📊 Results

- Achieved high accuracy in predicting loan repayment.
- Confusion matrix showed strong classification between repaid vs charged-off loans.
- Model generalized well with balanced precision and recall.

## ✅ Conclusion
This project demonstrates how deep learning can be applied to financial risk prediction.
Key takeaways:
- Borrower grades and subgrades strongly influence repayment likelihood.
- Employment length is not a reliable predictor.
- Proper preprocessing of categorical variables and handling missing values is critical.

By training on LendingClub data, the model can support financial institutions in better credit decision-making and minimizing risk exposure.
