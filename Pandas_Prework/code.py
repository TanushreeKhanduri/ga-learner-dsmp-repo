# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 
bank = pd.read_csv(path)
categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)
# code starts here
numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)




# code ends here


# --------------
# code starts here
banks = bank.drop(['Loan_ID'],axis=1)
#print(bank)
#banks = bank
#print(banks)
#print(banks.isnull().sum())
bank_mode = banks.mode()
print(bank_mode)
#banks = banks.fillna(bank_mode)
#banks = banks 
print("==============================")

#code ends here
for column in banks.columns:
    banks[column].fillna(banks[column].mode()[0], inplace=True)
print(banks.isnull().sum())


# --------------
# Code starts here
avg_loan_amount = pd.pivot_table(banks,index=['Gender', 'Married', 'Self_Employed'],values='LoanAmount')
print(avg_loan_amount)




# code ends here



# --------------
# code starts here
loanApprovedSECondition = (banks['Self_Employed'] == "Yes") & (banks['Loan_Status'] == "Y")
#loan_approved_se = banks[loanApprovedSECondition].count()
loan_approved_se = len(banks[loanApprovedSECondition])

loanApprovedNSECondition = (banks['Self_Employed'] == "No") & (banks['Loan_Status'] == "Y")
#loan_approved_nse = banks[loanApprovedNSECondition].count()
loan_approved_nse = len(banks[loanApprovedNSECondition])
print(loan_approved_se, loan_approved_nse)
Loan_Status  = 614
percentage_se = (loan_approved_se*100)/Loan_Status
percentage_nse = (loan_approved_nse*100)/Loan_Status

print(percentage_se)
print(percentage_nse)
# code ends here


# --------------
# code starts here
def mtoy(months):
    return months/12

loan_term = banks['Loan_Amount_Term'].apply(lambda x: mtoy(x))
print(loan_term)
def bigLoan(x):
    if(x>=25):
        return x
        


big_loan_term = (loan_term.apply(lambda x: bigLoan(x)))
big_loan_term = big_loan_term.dropna()
big_loan_term = len(big_loan_term)

#print(no_of_applicants)
print(big_loan_term)
#condition = banks['loan_term'] >= 25.0 
#big_loan_term = banks[condition]
#print(len(big_loan_term))

# code ends here


# --------------
# code starts here
loan_groupby = banks.groupby('Loan_Status')
loan_groupby = loan_groupby[['ApplicantIncome', 'Credit_History']]
mean_values = loan_groupby.mean()
print(mean_values)
# code ends here


