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
'trainSize' : 700,
'validationSize':15, 
'testSize' : 100,
'batchSize' :7,
'length' : 128,
'width' : 128,
'height' : 128,
'dataType' : 'float',
'numberOfParameter' : 3,
'numberOfParameterDigit' : 7,
'stopConsecutiveEpoch' : 1000,
'dataSetDir' : r'E:\NTNU1-2\Nyx\NyxDataSet\NyxDataSet128_128_128',
'startingTime' : startingTime,
'logDir' : 'E:\\NTNU1-2\\Nyx\\NyxDataSet\\wgan\\' + startingDate + '\\'
}

dataSet = {'Nyx':Nyx}