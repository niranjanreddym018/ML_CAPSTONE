"""Recreate SMALL, SYNTHETIC UI demos. Never use these for assignment claims."""
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
rng = np.random.default_rng(42)

n = 650
age = rng.integers(21, 69, n)
income = rng.integers(24000, 190000, n)
loan = rng.integers(3000, 105000, n)
credit = rng.integers(450, 800, n)
debt_ratio = rng.uniform(.05, .75, n)
bankruptcy = rng.binomial(1, .055, n)
defaults = rng.binomial(1, .09, n)
score = (50 + 13*debt_ratio + 9*bankruptcy + 7*defaults +
         .00006*loan - .00004*income - .019*(credit-600) + rng.normal(0,3.2,n))
loan_df = pd.DataFrame({
    'ApplicationDate':pd.date_range('2019-01-01',periods=n).astype(str),
    'Age':age,'AnnualIncome':income,'CreditScore':credit,
    'EmploymentStatus':rng.choice(['Employed','Self-Employed','Unemployed'],n,p=[.77,.16,.07]),
    'EducationLevel':rng.choice(['High School','Associate','Bachelor','Master','Doctorate'],n),
    'Experience':np.maximum(0,age-rng.integers(18,28,n)),'LoanAmount':loan,
    'LoanDuration':rng.choice([12,24,36,48,60],n),
    'MaritalStatus':rng.choice(['Single','Married','Divorced'],n),
    'NumberOfDependents':rng.integers(0,5,n),
    'HomeOwnershipStatus':rng.choice(['Own','Rent','Mortgage'],n),
    'MonthlyDebtPayments':rng.integers(80,2300,n),
    'CreditCardUtilizationRate':rng.uniform(.03,.95,n),
    'NumberOfOpenCreditLines':rng.integers(0,9,n),
    'NumberOfCreditInquiries':rng.integers(0,6,n),
    'DebtToIncomeRatio':debt_ratio,'BankruptcyHistory':bankruptcy,
    'LoanPurpose':rng.choice(['Home','Auto','Education','Debt Consolidation','Other'],n),
    'PreviousLoanDefaults':defaults,'PaymentHistory':rng.integers(0,35,n),
    'LengthOfCreditHistory':rng.integers(1,30,n),
    'SavingsAccountBalance':rng.integers(0,40000,n),
    'CheckingAccountBalance':rng.integers(0,30000,n),
    'TotalAssets':rng.integers(8000,900000,n),
    'TotalLiabilities':rng.integers(3000,230000,n),
    'MonthlyIncome':income/12,
    'UtilityBillsPaymentHistory':rng.uniform(0,1,n),
    'JobTenure':rng.integers(0,30,n),
    'NetWorth':rng.integers(0,800000,n),
    'BaseInterestRate':rng.uniform(.02,.12,n),
    'InterestRate':rng.uniform(.03,.19,n),
    'MonthlyLoanPayment':rng.integers(40,3400,n),
    'TotalDebtToIncomeRatio':rng.uniform(.1,1.2,n),
    'LoanApproved':rng.binomial(1,.55,n),
    'RiskScore':np.round(np.clip(score,0,100),2),
})
loan_df.loc[[3,13,36],"AnnualIncome"] = np.nan
loan_df.to_csv(root/'demo_credit_risk.csv',index=False)

n = 520
age = rng.integers(1,83,n)
glucose = rng.normal(112,30,n).clip(45,270)
bmi = rng.normal(28.5,6.0,n).clip(13,58)
hypertension = rng.binomial(1,.11,n)
heart = rng.binomial(1,.07,n)
logit = -6.1 + .052*age + .012*(glucose-100) + .65*hypertension + .8*heart
p = 1/(1+np.exp(-logit))
stroke = rng.binomial(1,p)
stroke_df = pd.DataFrame({
    'id':rng.choice(np.arange(10000,99000),n,replace=False),
    'gender':rng.choice(['Male','Female'],n),
    'age':age,'hypertension':hypertension,'heart_disease':heart,
    'ever_married':rng.choice(['Yes','No'],n),
    'work_type':rng.choice(['Private','Self-employed','Govt_job','children','Never_worked'],n),
    'Residence_type':rng.choice(['Urban','Rural'],n),
    'avg_glucose_level':np.round(glucose,2), 'bmi':np.round(bmi,1),
    'smoking_status':rng.choice(['never smoked','formerly smoked','smokes','Unknown'],n),
    'stroke':stroke,
})
stroke_df.loc[[8,12,19], 'bmi'] = np.nan
stroke_df.to_csv(root/'demo_stroke.csv',index=False)

n = 360
weight_class = rng.integers(0, 7, n)
weight = 48 + 12.5*weight_class + rng.normal(0,6,n)
height = rng.normal(1.7,.12,n).clip(1.4,2)
ob_df = pd.DataFrame({
    'Gender':rng.choice(['Female','Male'],n),
    'Age':rng.integers(17,63,n),'Height':np.round(height,3),
    'Weight':np.round(weight,2),
    'family_history_with_overweight':rng.choice(['yes','no'],n),
    'FAVC':rng.choice(['yes','no'],n),
    'FCVC':rng.uniform(1,3,n).round(3),
    'NCP':rng.uniform(1,4,n).round(3),
    'CAEC':rng.choice(['no','Sometimes','Frequently','Always'],n),
    'SMOKE':rng.choice(['yes','no'],n,p=[.07,.93]),
    'CH2O':rng.uniform(.6,3,n).round(3),
    'SCC':rng.choice(['yes','no'],n),
    'FAF':rng.uniform(0,3,n).round(3),
    'TUE':rng.uniform(0,2,n).round(3),
    'CALC':rng.choice(['no','Sometimes','Frequently','Always'],n),
    'MTRANS':rng.choice(['Public_Transportation','Walking','Automobile','Bike','Motorbike'],n),
    'NObeyesdad':np.array(['Insufficient_Weight','Normal_Weight','Overweight_Level_I',
                            'Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'])[weight_class],
})
ob_df.to_csv(root/'demo_obesity.csv',index=False)
print('Wrote synthetic samples:',len(loan_df),len(stroke_df),len(ob_df))
