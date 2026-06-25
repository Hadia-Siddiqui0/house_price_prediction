from linear_model import x_train,x_test,y_test,y_train,lr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred=lr.predict(x_test)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print(f'mae={mae}')
print(f'mse={mse}')
print(f'rmse={rmse}')
print(f'r2={r2}')

# Model Limitations
# - Outliers detected in price column (RMSE >> MAE)
# - Removing outliers could improve model performance
# - Left as future improvement