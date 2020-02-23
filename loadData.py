import numpy as np
import os 


def load_data(dirctory, dataSize):
    training_data = np.zeros((dataSize,16, 16, 1))
    g = os.walk(dirctory)  # r"E:\NyxOutput"
    flag = 0
    print('Start loading data...')
    for path,dir_list,file_list in g: 
        if flag >= dataSize:
            break
        if path.split('\\')[-1] == 'Raw_plt256_00200':
            try:
                print(f'File {flag}', end = '\r')
                #print('There are raw data in ', path)
                load_data = np.fromfile(path + '\\density.bin', dtype = 'float32').reshape((256, 256, 256))
                flag += 1
                load_data_16x16 = load_data[120:136, 120:136, 128]
                training_data[flag] = load_data_16x16
                #load_data.tofile(path.split('\\')[2]+'.bin')
            except:
                #print(f'No raw data in {path}')
                pass
    print(f'{dataSize} data have been loaded.')
    return training_data            

       
#load_data(r"E:\NyxOutput", dataSize=1000)



'''
for path,dir_list,file_list in g:  
    for file_name in file_list:  
        #print(file_name[5:12]+file_name[13:20] + file_name[21:28])
        print(file_name)
        tmp_data = np.fromfile(path+'\\'+file_name, dtype = 'float32')
        tmp_data_256_256_256 = tmp_data.reshape(256,256,256)
        tmp_data_16_16 = tmp_data_256_256_256[120:136, 120:136, 128]
        tmp_data_16_16.tofile(f'C:\\Users\\User\\Desktop\\Nyx\\GAN_Trainingset16_16\\16_16_{file_name}')
        parameter1 = float(file_name[5:12])
        parameter2 = float(file_name[13:20])
        parameter3 = float(file_name[21:28])
        (training_sample, training_label) = (tmp_data_16_16, (parameter1, parameter2, parameter3))
        training_set.append((training_sample, training_label))
print(len(training_set))

print(training_set[0])
print(training_set.shape)
print(training_set[0].shape)
cv2.imwrite('test', training_set[0])
'''