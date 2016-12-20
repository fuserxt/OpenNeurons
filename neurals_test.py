from pybrain.datasets            import ClassificationDataSet
from pybrain.datasets            import SupervisedDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure.modules   import TanhLayer
from pybrain.structure.modules   import LSTMLayer
from pybrain.structure.modules   import LinearLayer
from pybrain.structure.modules   import SigmoidLayer
from pybrain.structure           import FullConnection

from keras.models                import Sequential
from keras.layers                import Dense
from keras.layers                import Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils                 import np_utils

from sqlalchemy                  import create_engine

from sklearn.cross_validation    import train_test_split

from sklearn.model_selection     import cross_val_score
from sklearn.model_selection     import KFold
from sklearn.preprocessing       import LabelEncoder
from sklearn.pipeline            import Pipeline
from sklearn                     import datasets

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import seaborn           as sns
import requests

import psycopg2


class ETL:
    def __init__(self, db_name, user='postgres', host='localhost'):
        self.user = user
        self.host = host
        self.db_name = db_name
        self.engine = create_engine(
            'postgresql://{user}@{host}:5432/{db_name}'.format(user=user, host=host, db_name=db_name)
        )

    def write_to_db(self, path, table_name):
        extension = path.split('.')[-1]
        if extension == 'csv':
            data = pd.read_csv(path, encoding="ISO-8859-1", header=2)
        elif extension == 'xlsx':
            data = pd.read_excel(path, sheetname=2)
        data.to_sql(table_name, self.engine, if_exists='append')
    
    def get_table(self, table_name):
        return pd.read_sql_table(table_name, self.engine)
    
    def get_cb_data(self, url, path):
        '''
        XML is a tree-like structure, while a Pandas DataFrame
        is a 2D table-like structure. So there is no automatic way
        to convert between the two. You have to understand the XML
        structure and know how you want to map its data onto a 2D table,
        so now, when we don't know which xml is needed we would return it
        as text and save to file
        '''
        
        r = requests.get(url)
        with open(path, 'w') as file:
            file.write(r.text)
            
        return r.text        


def one_hot_encode_object_array(arr):
    encoder = LabelEncoder()
    encoder_arr = encoder.fit_transform(arr)
    return np_utils.to_categorical(encoder_arr)

def convert_supervised_to_classification(data, target):
    num_tr = data.shape[0]

    traindata = ClassificationDataSet(4,1,nb_classes=3)
    for i in range(num_tr):
        traindata.addSample(data[i], target[i])
       
    traindata._convertToOneOfMany()
    return traindata

def net_model_keras():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
    model.add(Dense(3, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def net_model_pybrain():
    network = FeedForwardNetwork()
    inLayer = LinearLayer(4) 
    hiddenLayer = SigmoidLayer(1)
    outLayer = SigmoidLayer(3) 
 
    network.addInputModule(inLayer) 
    network.addModule(hiddenLayer)
    network.addOutputModule(outLayer)

    in_to_hidden = FullConnection(inLayer , hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer , outLayer) 

    network.addConnection(in_to_hidden)
    network.addConnection(hidden_to_out)

    network.sortModules()

etl = ETL(db_name='opentrm')
etl.write_to_db('rb_e20161027.txt.csv', 'rb')
etl.write_to_db('Kospi Quotes Eikon Loader.xlsx', 'kospi')
csv_data = etl.get_cb_data('http://www.cbr.ru/scripts/XML_daily_eng.asp?date_req=02/03/2002', 'cb.xml')

etl.write_to_db("iris.data.txt.csv", 'iris')

data = etl.get_table('iris')
data.drop("index", inplace=True, axis=1)
data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

iris = datasets.load_iris()

target = data.species
data   = data.drop('species', axis=1)

test_X, train_X, test_y, train_y = train_test_split(data, target, train_size=0.0, random_state=42)

dummy_train_y = one_hot_encode_object_array(train_y)

#Keros
estimator = KerasClassifier(build_fn=net_model_keras, nb_epoch=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, train_X.values, dummy_train_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#PyBrain
convert_supervised_to_classification(iris.data, iris.target)
network = buildNetwork(4,1,3,outclass=SoftmaxLayer)
trainer = BackpropTrainer(network, dataset=traindata, momentum=0.1, verbose=True)
