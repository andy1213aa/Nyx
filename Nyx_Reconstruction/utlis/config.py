import datetime

# logfile dirs name by time
#============================
startingTime=datetime.datetime.now()
startingDate = f'{startingTime.year}_{startingTime.month}_{startingTime.day}'+'_'+f'{startingTime.hour}_{startingTime.minute}'
#============================

Nyx={
'dataSet' : 'Nyx',
'epochs' : 500,
'datasize' : 799,
'trainSize' : 699,
'validationSize':59, 
'testSize' : 100,
'batchSize' :16,
'length' : 64,
'width' : 64,
'height' : 64,
'dataType' : 'float',
'numberOfParameter' : 3,
'numberOfParameterDigit' : 7,
'stopConsecutiveEpoch' : 100,
'dataSetDir' :  '/home/csun001/Nyx/NyxDataSet/Nyx_tfrecords/NyxDataSet64_64_64.tfrecords',
'startingTime' : startingTime,
'logDir' : '/home/csun001/Nyx/NyxDataSet/log/Nyx_' + startingDate + '/'
}

dataSet = {'Nyx':Nyx}