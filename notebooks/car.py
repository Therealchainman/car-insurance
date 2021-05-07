import nni
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import numpy
import pandas

class SendMetrics(Callback):
    """
    Keras callback to send metrics to NNI framework
    """
    def on_epoch_end(self, epoch, logs=None):
        """
        run on end of each epoch
        """
        nni.report_intermediate_result(logs['accuracy'])

def buildModel(args):
    df_train = pandas.read_csv('/data/projects/car-insurance-ssh/data/df_train.csv')
    X_train, y_train = numpy.array(df_train.drop(columns=['car_insurance'])), numpy.array(df_train.car_insurance)
    n_shape = X_train.shape[1]
    model = Sequential()
    adam = Adam(lr=args['lr'])
    model.add(Dense(4, input_shape=(n_shape,),activation='softplus'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(X_train,y_train,epochs=args['epochs'],batch_size=args['batch_size'],verbose=0, callbacks=[SendMetrics()])
    
if __name__ == '__main__':
    params = {'batch_size':32, 'lr': 0.001, 'epochs':3}
    params = nni.get_next_parameter()
    buildModel(params)