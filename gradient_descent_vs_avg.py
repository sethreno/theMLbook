# predects sales based on radio advertising using simple average
# I created this because I wanted to compare the results with the gradient descent method
import numpy as np

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

total_radio_spend = np.sum(x)
total_sales = np.sum(y)
sales_per_radio_spend = total_sales / total_radio_spend

x_new = 23.0
y_new = x_new * sales_per_radio_spend
print(y_new)
