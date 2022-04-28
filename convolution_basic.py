# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def conv1():
    f1 = lambda t: np.maximum(0, 1 - abs(t))
    f2 = lambda t: (t > 0) * np.exp(-2 * t)
    Fs = 50  # our sampling frequency for the plotting
    T = 5  # the time range we are interested in
    t = np.arange(-T, T, 1 / Fs)  # the time samples
    plt.plot(t, f1(t), label='$f_1(t)$')
    plt.plot(t, f2(t), label='$f_2(t)$')

    plt.grid()
    plt.legend(loc="upper left")
    plt.show()

def conv2():
    f1 = lambda t: np.maximum(0, 1 - abs(t))
    f2 = lambda t: (t > 0) * np.exp(-2 * t)
    Fs = 80  # our sampling frequency for the plotting
    T = 2  # the time range we are interested in
    t = np.arange(-T, T, 1 / Fs)  # the time samples

    t0 = 1
    flipped = lambda tau: f2(t0 - tau)
    product = lambda tau: f1(tau) * f2(t0 - tau)

    plt.figure(figsize=(8, 3))
    plt.plot(t, f1(t), label=r'$f_1(\tau)$')
    plt.plot(t, flipped(t), label=r'$f_2(t_0-\tau)$')
    plt.plot(t, product(t), label=r'$f_1(\tau)f_2(t_0-\tau)$')

    plt.grid()
    plt.legend(loc="upper left")
    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    conv2()




