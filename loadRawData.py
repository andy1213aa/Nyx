import  numpy as np
import os 

def loadData(directory, width, length, height):
    g = os.walk(directory)
    flag = 0
    for path, dirList, fileList in g:
        trainingData = np.zeros((len(fileList), width, length, height))
        print(f'You have {len(fileList)} data.')     
        for data in fileList:
            print(f'Starting Loging data {flag}', end='\r')
            load_data = np.fromfile(directory + '\\' + data, dtype = 'float32').reshape((256, 256, 256))
            
            load_data_16x16 = load_data[120:136, 120:136, 128:129]
            
            trainingData[flag] = load_data_16x16
            flag += 1
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
