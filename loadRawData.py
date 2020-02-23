import  numpy as np
import os 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
def loadData(directory, width, length, height):
    g = os.walk(directory)
    flag = 0
    for path, dirList, fileList in g:
        trainingData = np.zeros((30, width, length, height))
        print(f'You have {len(fileList)} data.')     
        for data in fileList:
            xIndex = random.randint(0, 240)
            yIndex = random.randint(0, 240)
            zIndex = random.randint(0, 240)
            print(f'Data {flag}: ', xIndex, yIndex, zIndex)
            print(f'Starting Loging data {flag}', end='\r')
            load_data = np.fromfile(directory + '\\' + data, dtype = 'float32').reshape((256, 256, 256))
            
            load_data_16x16 = load_data[xIndex:xIndex+16, yIndex:yIndex+16, zIndex:zIndex+1]
            
            trainingData[flag] = load_data_16x16
            flag += 1
            if flag == 30:
                break
    print('')
    print('Done!')
    
    return trainingData
def plot3D(result):
    for i in result:
        fig = plt.figure()
        axis = plt.axes(projection='3d')
        x = range(0, 16)
        y= range(0, 16)
        xData, yData = np.meshgrid(x, y)
        zData = i[x, y]
        surface = axis.plot_surface(xData, yData, zData, rstride=1, cstride=1, cmap='coolwarm_r')
        fig.colorbar(surface, shrink=1.0, aspect=20)
        plt.show()
result = loadData(r'C:\Users\Andy\Desktop\Nyx\NyxDataSet', 16, 16, 1)
plot3D(result)




