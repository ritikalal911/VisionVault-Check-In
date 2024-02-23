from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


class Architecture():

    def __init__(self,input_shape,num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        model = Sequential()
        model.add(Dense(1024,activation='relu',input_shape=self.input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(1024,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes,activation='softmax'))

        model.compile(optimizer=Adam(lr=0.001),
                      loss=sparse_categorical_crossentropy,
                      metrics=['accuracy'])
        
        return model