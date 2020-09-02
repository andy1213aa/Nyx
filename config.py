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
'validationSize':15, 
'testSize' : 100,
'batchSize' :8,
'length' : 64,
'width' : 64,
'height' : 64,
'dataType' : 'float',
'numberOfParameter' : 3,
'numberOfParameterDigit' : 7,
'stopConsecutiveEpoch' : 100,
'dataSetDir' : r'E:\NTNU1-2\Nyx\NyxDataSet\NyxDataSet64_64_64',
'startingTime' : startingTime,
'logDir' : 'E:\\NTNU1-2\\Nyx\\NyxDataSet\\wgan\\' + startingDate + '\\'
}

dataSet = {'Nyx':Nyx}