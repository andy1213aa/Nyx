import  numpy as np
import os 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def loadData(directory, width, length, height):
    g = os.walk(directory)
    flag = 0
    for path, dirList, fileList in g:
        trainingData = np.zeros((30, width, length, height))
        print(f'You have {len(fileList)} data.')     
        for data in fileList:
            print(f'Starting Loging data {flag}', end='\r')
            load_data = np.fromfile(directory + '\\' + data, dtype = 'float32').reshape((256, 256, 256))
            
            load_data_16x16 = load_data[120:136, 120:136, 128:129]
            
            trainingData[flag] = load_data_16x16
            flag += 1
            if flag == 30:
                break
    print('')
    print('Done!')
    
    return trainingData
result = loadData(r'C:\Users\Andy\Desktop\Nyx\NyxDataSet', 16, 16, 1)
flag = 0

for i in result:
    print("Data: ", flag)
    print("Std: ", i.std())
    print("Mean: ", i.mean())
    flag += 1
    print(i.shape)
    fig = plt.figure()
    axis = plt.axes(projection='3d')
    x = range(0, 16)
    y= range(0, 16)
    xData, yData = np.meshgrid(x, y)
    zData = i[x, y]
    surface = axis.plot_surface(xData, yData, zData, rstride=1, cstride=1, cmap='coolwarm_r')
    fig.colorbar(surface, shrink=1.0, aspect=20)
    plt.show()


