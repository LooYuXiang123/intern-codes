from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
#xfrom scipy.interpolate import UnivariateSpline
from matplotlib import pyplot
import numpy as np
import sys

# py plotCurve.py 720_095.csv 720_050.csv 1080_095.csv 1080_050.csv

# Polynomial function to define general shape of curve
def objective(x, a, b, c):
	return a * x + b * x**2 + c
 
# Load the dataset
url1 = sys.argv[1]
url2 = sys.argv[2]
url3 = sys.argv[3]
url4 = sys.argv[4]



dataframe1 = read_csv(url1, header=None)
dataframe2 = read_csv(url2, header=None)
dataframe3 = read_csv(url3, header=None)
dataframe4 = read_csv(url4, header=None)


data1 = dataframe1.values
data2 = dataframe2.values
data3 = dataframe3.values
data4 = dataframe4.values


# Depends on layout of csv file
x1, y1 = data1[:, 1], data1[:, 0]
x2, y2 = data2[:, 1], data2[:, 0]
x3, y3 = data3[:, 1], data3[:, 0]
x4, y4 = data4[:, 1], data4[:, 0]
#x5, y5 = data4[:, 1], data4[:, 0]
#x6, y6 = data4[:, 1], data4[:, 0]

# curve fit (to determine best possible values for constants to give best fit curve)
#s1 = UnivariateSpline(x1, y1, s=50)
#xs1 = np.linspace(10, 45, 50)
#ys1 = s1(xs1)

#s2 = UnivariateSpline(x2, y2, s=50)
#xs2 = np.linspace(10, 45, 50)
#ys2 = s2(xs2)

#s3 = UnivariateSpline(x3, y3, s=50)
#xs3 = np.linspace(10, 45, 50)
#ys3 = s3(xs3)

#s4 = UnivariateSpline(x4, y4, s=50)
#xs4 = np.linspace(10, 45, 50)
#ys4 = s4(xs4)

#popt1, _ = curve_fit(objective, x1, y1)
#popt2, _ = curve_fit(objective, x2, y2)
#popt3, _ = curve_fit(objective, x3, y3)
#popt4, _ = curve_fit(objective, x4, y4)

# summarize the parameter values
#a, b, c = popt1
#d, e, f = popt2
#g, h, i = popt3
#j, k, l = popt4
#print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
#print('y = %.5f * x + %.5f * x^2 + %.5f' % (d, e, f))
#print('y = %.5f * x + %.5f * x^2 + %.5f' % (g, h, i))
#print('y = %.5f * x + %.5f * x^2 + %.5f' % (j, k, l))

# plot input vs output
#pyplot.scatter(x1, y1, c='red')
#pyplot.scatter(x2, y2, c='blue')
#pyplot.scatter(x3, y3, c='green')
#pyplot.scatter(x4, y4, c='black')

# define a sequence of inputs between the smallest and largest known inputs
#x_line1 = arange(min(x1), max(x1), 1)
# calculate the output for the range
#y_line1 = objective(x_line1, a, b, c)

# define a sequence of inputs between the smallest and largest known inputs
#x_line2 = arange(min(x2), max(x2), 1)
# calculate the output for the range
#y_line2 = objective(x_line2, d, e, f)

# define a sequence of inputs between the smallest and largest known inputs
#x_line3 = arange(min(x3), max(x3), 1)
# calculate the output for the range
#y_line3 = objective(x_line3, g, h, i)

# define a sequence of inputs between the smallest and largest known inputs
#x_line4 = arange(min(x4), max(x4), 1)
# calculate the output for the range
#y_line4 = objective(x_line4, j, k, l)

# create a line plot for the mapping function
# Figure 1
#pyplot.plot(xs1, ys1, '-', color='red', label='720p(0.95)')
#pyplot.plot(xs2, ys2, '-', color='blue', label='720p(0.50)')
#pyplot.plot(x_line1, y_line1, '-', color='red', label='720p(0.95)')
#pyplot.plot(x_line2, y_line2, '-', color='blue', label='720p(0.50)')
listOf_Xticks = np.arange(0, 50, 10)
listOf_Yticks = np.arange(0, 110, 10)
pyplot.scatter(x1, y1, c='red', label='720p(0.95)')
pyplot.scatter(x2, y2, c='blue', label='720p(0.50)')
pyplot.xticks(listOf_Xticks)
pyplot.yticks(listOf_Yticks)
pyplot.xlabel("Distance/m")
pyplot.ylabel("mAP/%")
pyplot.legend()
pyplot.figure()


# Figure 2
#pyplot.plot(xs3, ys3, '-', color='black', label='1080p(0.95)')
#pyplot.plot(xs4, ys4, '-', color='green', label='1080p(0.50)')
pyplot.scatter(x3, y3, c='green', label='1080p(0.95)')
pyplot.scatter(x4, y4, c='black', label='1080p(0.50)')
pyplot.xticks(listOf_Xticks)
pyplot.yticks(listOf_Yticks)
#pyplot.plot(x_line3, y_line3, '-', color='green', label='1080p(0.95)')
#pyplot.plot(x_line4, y_line4, '-', color='black', label='1080p(0.50)')
pyplot.xlabel("Distance/m")
pyplot.ylabel("mAP/%")
pyplot.legend()
pyplot.show()
