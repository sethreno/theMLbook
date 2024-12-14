import numpy as np
from sklearn.linear_model import LinearRegression

# constants for columns in data.txt
TV_COL = 1
RADIO_COL = 2
PAPER_COL = 3
SALES_COL = 4

x, y = np.loadtxt(
    "data.txt",
    skiprows=1,
    usecols=(RADIO_COL, SALES_COL),
    delimiter=",",
    unpack=True,
)

# produces the same results as gradient_descent.py
# but it's 2.8x faster and requires less code
model = LinearRegression().fit(x.reshape(-1, 1), y)

x_new = 23.0
y_new = model.predict(np.array([[x_new]]))
print(y_new[0])
