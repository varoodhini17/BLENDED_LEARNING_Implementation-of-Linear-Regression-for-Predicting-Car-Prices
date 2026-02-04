# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Import required libraries for data handling, visualization, and machine learning.

2.  Load the dataset CarPrice_Assignment.csv using Pandas.

3.  Select the independent variables (enginesize, horsepower, citympg, highwaympg) and the dependent variable (price).

4.  Split the dataset into training (80%) and testing (20%) sets.

5.  Apply StandardScaler to normalize the training and testing data.

6.  Create a Linear Regression model using LinearRegression().

7.  Train the model using the scaled training data.

8.  Predict car prices using the trained model on test data.

9.  Evaluate the model using MSE, MAE, RMSE, and R-squared metrics.

10. Analyze model assumptions using plots and statistical tests (Actual vs Predicted, residual plots, Durbin–Watson test, and Q-Q plot)

## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
df=pd.read_csv('CarPrice_Assignment.csv')
df.head()

X = df[['enginesize','horsepower','citympg','highwaympg']]
Y = df['price']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled, Y_train)
Y_pred = model.predict(X_test_scaled)

print('Name:Varoodhini M')
print('Reg. No:212225220118')
print("MODEL COEFFICIENTS:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:>12}:{coef:>10}")
print(f"{'Intercept':>12}:{model.intercept_:>10}")

print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}:{mean_squared_error(Y_test,Y_pred):>10.2f}")
print(f"{'RMSE':>12}:{np.sqrt(mean_squared_error(Y_test,Y_pred)):>10.2f}")
print(f"{'R-squared':>12}: {r2_score(Y_test,Y_pred):>10.2f}")
plt.figure(figsize=(10,5))
plt.scatter(Y_test,Y_pred, alpha=0.6)
plt.plot([Y.min(),Y.max()],[Y.min(),Y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()
residuals =Y_test-Y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}",
     "\n(Values close to 2 indicate no autocorrelation)")
plt.figure(figsize=(10,5))
sns.residplot(x=Y_pred,y=residuals,lowess=True,line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals vs Precdicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residual ($)")
plt.grid(True)
plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals,kde=True,ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals,line='45',fit=True,ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()


```

## Output:
## Linearity check: actual vs predicted prices
<Figure size 1000x500 with 1 Axes><img width="868" height="468" alt="image" src="https://github.com/user-attachments/assets/c54b87c8-f1b2-4abb-9490-115078947970" />

## Homoscedasticity check: actual vs predicted
<Figure size 1000x500 with 1 Axes><img width="880" height="468" alt="image" src="https://github.com/user-attachments/assets/6acf3794-f311-44b4-b277-3c635d320fa6" />

 ## Residuals distribution and Q-Q Plot
<Figure size 1200x500 with 2 Axes><img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/22380bb4-e0cc-428c-aed8-0c8c75577ca8" />






## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
