import numpy as np
import matplotlib.pyplot as plt


def estimate_coefficient(x,y):
     # no of observation
     n = np.size(x)

     #calcualting cross deviation  and deviatiom
     ss_xy1 = np.sum(y*x)

     #mean of x and y
     m_x, m_y = np.mean(x), np.mean(y)

     #calculating the cross deviation and deviaation of x
     ss_xy = np.sum(y*x) - n*m_y*m_x
     ss_xx = np.sum(x * x) - n * m_x * m_x

     #calculating regression cofficient
     b_1 = ss_xy / ss_xx
     b_0 = m_y - b_1*m_x

     return b_1, b_0


def plot_regression_line(x, y, b):
    #plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m", marker ="o", s=30)

    #predicted response vector
    y_pred = b[0] + b[1]*x

    #plotting the regression line
    plt.plot(x, y_pred, color ="g")

    #putting labels
    plt.xlabel("x")
    plt.ylabel("y")

    #function to show plot
    plt.show()


def main():
     #observation
     x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
     y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

     #estimate cofficient
     b = estimate_coefficient(x,y)

     #plotting regression line
     plot_regression_line(x, y, b)


if __name__ == "__main__":
     main()
