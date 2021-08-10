from model_merger.keras.proc.calculator import LCaliculator, calc_l1_norm
from keras.models import Model
from keras.layers import Input
from model_merger.keras.proc.checkpoint import BaseModelCheckPointer


class SiameseBuilder(object):

    def __init__(self,
                 base_model: Model,
                 monitor='val_loss',
                 mode='auto'):
        self.__base_model = base_model
        self.__monitor = monitor
        self.__mode = mode

    @property
    def output_shape(self):
        return self.__base_model.output_shape

    @property
    def output_layer(self):
        return self.__base_model.output

    @property
    def teacher_input_layer(self):
        return Input(shape=tuple(self.__base_model.output_shape[1:]))

    @property
    def input_layer(self):
        return Input(batch_shape=self.__base_model.input_shape)

    def build_input_teacher_from_output(self):
        return Input(shape=self.output_shape)

    def build_shame_trainer_for_classifivation(self,
                                               q: float,
                                               optimizer,
                                               loss_func=None,
                                               calc_distance=calc_l1_norm,
                                               filepath=None,
                                               save_best_only=True,
                                               save_weights_only=False):
        calculator = LCaliculator(q, loss_func, calc_distance)
        inputs = [self.input_layer, self.input_layer]
        predict_outputs = [self.__base_model(input_batch) for input_batch in inputs]
        output_loss = calculator.build_loss_layer()(predict_outputs)
        train_model = Model(inputs=inputs, outputs=output_loss)
        train_model.compile(optimizer=optimizer, loss=calculator.loss_func, metrics=['accuracy'])
        train_model.summary()
        if filepath is not None:
            base_model_checkpoint = BaseModelCheckPointer(self.__base_model,
                                                          filepath,
                                                          self.__monitor,
                                                          save_best_only,
                                                          save_weights_only,
                                                          self.__mode)
            return train_model, [base_model_checkpoint]

        def build_checkpoint(base_filepath):
            return BaseModelCheckPointer(self.__base_model,
                                         base_filepath,
                                         self.__monitor,
                                         save_best_only,
                                         save_weights_only,
                                         self.__mode)
        return train_model, build_checkpoint


