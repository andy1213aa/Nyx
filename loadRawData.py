import  numpy as np
import os 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def loadData(directory, width, length, height):
    g = os.walk(directory)
    flag = 0
    for path, dirList, fileList in g:
        trainingData = np.zeros((len(fileList), width, length, height))
        print(f'You have {len(fileList)} data.')     
        for data in fileList:
            print(f'Starting Loging data {flag}', end='\r')
            load_data = np.fromfile(directory + '\\' + data, dtype = 'float32').reshape((256, 256, 256))
            
            trainingData[flag] = load_data[120:136, 120:136, 120:136]
            flag += 1
            # if flag == 30:
            #     break
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
#result = loadData(r'C:\Users\User\Desktop\NTNU 1-2\Nyx\NyxDataSet', 16, 16)

# start = 0

# iteration = 10
# for step in range(iteration):
#     batch = result[start: start+3]
#     print('batch size: ', batch.shape)
#     for i in range(16):
#         mean = np.mean(batch[:, i, :, 0])
#         std = np.std(batch[:, i, :, 0])
#         print(batch[:, i, :, 0])
#         batch[:, i, :, 0] = (batch[:, i, :, 0]-mean) / std
#         print(batch[:, i, :, 0])
#         break
#     break
    #print(f'{i} std: ', np.std(i))


