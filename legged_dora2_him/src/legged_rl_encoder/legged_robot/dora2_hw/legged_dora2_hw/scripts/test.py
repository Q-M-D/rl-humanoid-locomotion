import matplotlib.pyplot as plt
import pandas as pd

l1 = [1, 2, 3, 4]
l2 = [2, 4, 6, 8]

def func():
    fig, ax = plt.subplots()
    ax.plot(l1, l2)
    plt.show(block=False)
    plt.pause(1)
    input("press any key to break...")
    plt.close(fig)

while True:
    func()