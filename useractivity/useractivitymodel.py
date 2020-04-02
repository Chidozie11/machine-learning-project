import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pressure_left(dataset):
    plt.figure()
    start = 321
    for i in range(len(dataset)):
        plt.subplot(start)
        data = dataset[i][:5000]
        plt.title(data.activity[0])
        data = data.set_index(['time'])
        plt.plot(data.index, data['PL1'], label='PL1')
        plt.plot(data.index, data['PL2'], label='PL2')
        plt.plot(data.index, data['PL3'], label='PL3')
        plt.plot(data.index, data['PL4'], label='PL4')
        plt.plot(data.index, data['PL5'], label='PL5')
        plt.plot(data.index, data['PL6'], label='PL6')
        plt.plot(data.index, data['PL7'], label='PL7')
        plt.plot(data.index, data['PL8'], label='PL8')
        plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.gca().xaxis.set_visible(False)
        start = start + 1

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()


def plot_acc(dataset):
    plt.figure()
    start = 321
    for i in range(len(dataset)):
        plt.subplot(start)
        data = dataset[i][:5000]
        plt.title(data.activity[0])
        data = data.set_index(['time'])
        plt.plot(data.index, data['AX'], label='AX')
        plt.plot(data.index, data['AY'], label='AY')
        plt.plot(data.index, data['AZ'], label='AZ')
        plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.gca().xaxis.set_visible(False)
        start = start + 1

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()


def plot_gyro(dataset):
    plt.figure()
    start = 321
    for i in range(len(dataset)):
        plt.subplot(start)
        data = dataset[i][:2000]
        plt.title(data.activity[0])
        data = data.set_index(['time'])
        plt.plot(data.index, data['GX'], label='GX')
        plt.plot(data.index, data['GY'], label='GY')
        plt.plot(data.index, data['GZ'], label='GZ')
        plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.gca().xaxis.set_visible(False)
        start = start + 1

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()


def plot_pressure_right(dataset):
    plt.figure()
    start = 321
    for i in range(len(dataset)):
        plt.subplot(start)
        data = dataset[i][:3000]
        plt.title(data.activity[0])
        data = data.set_index(['time'])
        plt.plot(data.index, data['PR1'], label='PR1')
        plt.plot(data.index, data['PR2'], label='PR2')
        plt.plot(data.index, data['PR3'], label='PR3')
        plt.plot(data.index, data['PR4'], label='PR4')
        plt.plot(data.index, data['PR5'], label='PR5')
        plt.plot(data.index, data['PR6'], label='PR6')
        plt.plot(data.index, data['PR7'], label='PR7')
        plt.plot(data.index, data['PR8'], label='PR8')
        plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.gca().xaxis.set_visible(False)
        start = start + 1

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()

