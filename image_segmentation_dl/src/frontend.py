
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import adam

from src.Datagen import DataSequence
from src.backend import ENET, VGG, UNET
from keras import callbacks

class Segment(object):
    def __init__(self, backend,
                 input_size, nb_classes):
        self.input_size = input_size
        self.nb_classes = nb_classes
        if backend == "ENET":
            self.feature_extractor = ENET(self.input_size, self.nb_classes).build()
        elif backend == "VGG":
            self.feature_extractor = VGG(self.input_size, self.nb_classes).build()
        elif backend == "UNET":
            self.feature_extractor = UNET(self.input_size, self.nb_classes).build()
        else:
            raise ValueError('only support three models now')

    def train(self, train_configs):
        train_times = train_configs["train_times"]

        # taking batch of data for batch gradient descent
        sequence = DataSequence(train_configs["data_directory"] + "data_road/training", train_configs["batch_size"],
                                self.input_size)
        steps_per_epoch = len(sequence) * train_times

        # optimizer and loss function selections.
        optimizer = adam(train_configs["learning_rate"])
        self.feature_extractor.compile(optimizer=optimizer, loss='categorical_crossentropy',
                                       metrics=['accuracy'])

        # define the callbacks for training
        tb = TensorBoard(log_dir=train_configs["logs_dir"], write_graph=True)
        mc = ModelCheckpoint(mode='max', filepath=train_configs["save_model_name"], monitor='acc',
                             save_best_only='True',
                             save_weights_only='True', verbose=2)
        es = EarlyStopping(mode='max', monitor='acc', patience=6, verbose=1)
        model_reducelr = callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.2,
            patience=5,
            verbose=1,
            min_lr=0.05 * train_configs["learning_rate"])

        callback = [tb, mc, es, model_reducelr]

        # starting training
        self.feature_extractor.fit_generator(sequence,
                                             steps_per_epoch=steps_per_epoch,
                                             epochs=train_configs["nb_epochs"],
                                             verbose=1,
                                             shuffle=True, callbacks=callback,
                                             workers=3,
                                             max_queue_size=8
                                             )
