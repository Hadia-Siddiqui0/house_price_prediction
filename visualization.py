import matplotlib.pyplot as plt
import seaborn as sns
from train_test_split import x_train,x_test,y_train,y_test
from linear_model import lr
import pandas as pd


# scatter plot of predicted and actual value
y_pred = lr.predict(x_test)

plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.show()

# residual plot to check homoscedacity
residual=y_test-y_pred

plt.scatter(y_pred,residual,alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

#  visualizing which feature is most important
coef_df = pd.DataFrame({
    'Feature': x_train.columns,
    'Coefficient': lr.coef_
}).sort_values('Coefficient', ascending=False)

sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.title("Feature Importance")
plt.show()