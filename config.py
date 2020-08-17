import datetime

# logfile dirs name by time
#============================
startingTime=datetime.datetime.now()
startingDate = f'{startingTime.year}-{startingTime.month}-{startingTime.day}'+'_'+f'{startingTime.hour}-{startingTime.minute}'
#============================

Nyx={
'dataSet' : 'Nyx',
'epochs' : 5000,
'datasize' : 799,
'trainSize' : 699,
'validationSize': 59, 
'testSize' : 100,
'batchSize' : 64,
'length' : 16,
'width' : 16,
'height' : 16,
'dataType' : 'float',
'numberOfParameter' : 3,
'numberOfParameterDigit' : 7,
'stopConsecutiveEpoch' : 500,
'dataSetDir' : r'E:\NTNU1-2\Nyx\NyxDataSet\NyxDataSet16_16_16',
'startingTime' : startingTime,
'logDir' : 'E:\\NTNU1-2\\Nyx\\NyxDataSet\\wgan\\' + startingDate + '\\'

}

dataSet = {'Nyx':Nyx}