from keras.layers import Lambda, Activation, Input
from keras.losses import binary_crossentropy
import keras.engine.training
from keras.optimizers import Optimizer, SGD
from keras.models import Model
from network_model.build_model import builder as base_model_builder
from typing import Optional, List, Union, Tuple, Callable
from network_model.model_for_distillation import ModelForDistillation
from network_model.model_base.tempload import builder as temp_loader


DistllationModelIncubator = Callable[[str, List[str]], ModelForDistillation]


def fix_weight(model):
    print(model)
    for layer in model.layers:
        layer.trainable = False
    return model


class DistllationModelBuilder(object):

    def __init__(self, temperature: float, loss_lambda: float, output_activation: str = "softmax"):
        self.__temperature = temperature
        self.__lambda = loss_lambda
        self.__output_activation = output_activation

    @property
    def temperature(self):
        return self.__temperature

    @property
    def loss_lambda(self):
        return self.__lambda

    @property
    def logits_layer_builder(self):
        return Lambda(lambda x: x/self.temperature)

    @property
    def output_activation(self):
        return self.__output_activation

    def loss_part_of_true(self, y_true, y_pred):
        return (1-self.loss_lambda)*binary_crossentropy(y_true, y_pred)

    def loss_part_of_soft(self, y_soft, y_pred_soft):
        return self.loss_lambda*self.temperature*binary_crossentropy(y_soft, y_pred_soft)

    def knowledge_distillation_loss(self, input_distillation):
        y_pred, y_true, y_soft, y_pred_soft = input_distillation
        return self.loss_part_of_true(y_true, y_pred) + self.loss_part_of_soft(y_soft, y_pred_soft)

    @property
    def build_self_output_loss_func_layer(self):
        return Lambda(self.knowledge_distillation_loss, output_shape=(1,), name='kd_')

    def build_teacher_model(self,
                            teacher_base: keras.engine.training.Model,
                            loss: str = 'categorical_crossentropy',
                            optimizer: Optimizer = SGD()):
        teacher_model = fix_weight(teacher_base)
        teacher_model.compile(optimizer=optimizer, loss=loss)
        #teacher_model.layers.pop()
        theacher_logits = teacher_model.layers[-1].output
        theacher_logits_t = self.logits_layer_builder(theacher_logits)
        teacher_probabilities_t = Activation(self.output_activation)(theacher_logits_t)
        return teacher_model, teacher_probabilities_t

    def build_student_model(self,
                            input_layer,
                            student_base: keras.engine.training.Model,
                            loss: str = 'categorical_crossentropy',
                            optimizer: Optimizer = SGD()):
        output_layer = student_base(input_layer)
        print('student')
        student_base.summary()
        pre_output_layer = student_base.layers[-1].output
        logits_t = self.logits_layer_builder(pre_output_layer)
        probabilities_t = Activation(self.output_activation, name="probabilities_T")(logits_t)
        student_model = Model(inputs=[input_layer], outputs=[output_layer])
        student_model.compile(optimizer=optimizer, loss=loss, metrics=['mean_iou'])
        student_model.summary()
        return student_model, probabilities_t

    def build_raw_model(self,
                        teacher_base: keras.engine.training.Model,
                        student_base: keras.engine.training.Model,
                        teacher_optimizer: Optimizer = SGD(),
                        student_optimizer: Optimizer = SGD(),
                        loss: str = 'categorical_crossentropy'):
        teacher_model, teacher_probabilities_t = self.build_teacher_model(teacher_base, loss, teacher_optimizer)
        input_layer = teacher_model.input
        student_model, probabilities_t = self.build_student_model(input_layer, student_base, loss, student_optimizer)
        output = student_model.output
        input_true = Input(name='input_true', shape=[None], dtype='float32')
        inputs = [input_layer, input_true]
        output_loss = self.build_self_output_loss_func_layer([output,
                                                              input_true,
                                                              teacher_probabilities_t,
                                                              probabilities_t])
        train_model = Model(inputs=inputs, outputs=output_loss)
        return train_model, student_model

    def build_model_builder(self,
                            teacher_base: keras.engine.training.Model,
                            teacher_optimizer: Optimizer = SGD(),
                            student_optimizer: Optimizer = SGD(),
                            loss: str = 'categorical_crossentropy',
                            callbacks: Optional[List[keras.callbacks.Callback]] = None,
                            monitor: str = "",
                            will_save_h5: bool = True) -> DistllationModelIncubator:
        class_num = teacher_base.output_shape[-1]
        input_shape = tuple(teacher_base.input_shape[1:])
        img_size = input_shape[0]
        channel = input_shape[-1]

        def load_model_from_name(student_model_name: str, class_set: List[str]) -> ModelForDistillation:
            student_base = base_model_builder(class_num=class_num,
                                              img_size=img_size,
                                              channels=channel,
                                              optimizer=student_optimizer,
                                              model_name=student_model_name)
            train_model, student_model = self.build_raw_model(teacher_base=teacher_base,
                                                              student_base=student_base,
                                                              teacher_optimizer=teacher_optimizer,
                                                              student_optimizer=student_optimizer,
                                                              loss=loss)
            return ModelForDistillation(train_model=train_model,
                                        student_model=student_model,
                                        class_set=class_set,
                                        callbacks=callbacks,
                                        monitor=monitor,
                                        will_save_h5=will_save_h5)
        return load_model_from_name

    def build_model_builder_from_teacher_files(self,
                                               teacher_bases_param: Union[str, Tuple[str, str]],
                                               teacher_optimizer: Optimizer = SGD(),
                                               student_optimizer: Optimizer = SGD(),
                                               loss: str = 'categorical_crossentropy',
                                               callbacks: Optional[List[keras.callbacks.Callback]] = None,
                                               monitor: str = "",
                                               will_save_h5: bool = True) -> DistllationModelIncubator:
        teacher_base = temp_loader(temp_paths=teacher_bases_param)
        return self.build_model_builder(teacher_base,
                                        teacher_optimizer,
                                        student_optimizer,
                                        loss,
                                        callbacks,
                                        monitor,
                                        will_save_h5)

