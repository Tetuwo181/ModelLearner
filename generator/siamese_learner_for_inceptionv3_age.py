from generator.siamese_learner import SiameseLearnerDataBuilder,  build_other_batch, build_siamese_labels_for_space
import numpy as np
from generator.transpose import transpose
from abc import ABC, abstractmethod
from typing import Optional


class TeacherPreprocessor(ABC):

    def __init__(self, age_divide_index):
        self.__age_divide_index = age_divide_index

    @abstractmethod
    def decide_aux(self, label):
        pass

    @abstractmethod
    def preprocess_main(self, label):
        pass

    def run_preprpocess(self, teacher):
        return [(self.preprocess_main(label), self.decide_aux(label)) for label in teacher]

    def build_main_teacher(self, base_teacher):
        return np.array([self.preprocess_main(label) for label in base_teacher])

    def build_aux_teacher(self, base_teacher):
        return np.array([self.decide_aux(label) for label in base_teacher])

    def build_teacher_for_train(self, teacher):
        return self.build_main_teacher(teacher), self.build_aux_teacher(teacher)


class SiameseLearnerDataBuilderForInceptionV3(SiameseLearnerDataBuilder):

    def __init__(self,
                 will_transpose: bool,
                 convert_numpy: bool,
                 build_set_num: int,
                 margin: int,
                 teacher_preprocessor: TeacherPreprocessor,
                 aux_margin: Optional[int] = None):
        super(SiameseLearnerDataBuilderForInceptionV3, self).__init__(will_transpose,
                                                                      convert_numpy,
                                                                      build_set_num,
                                                                      margin,
                                                                      aux_margin)
        self.__teacher_preprocessor = teacher_preprocessor

    def __call__(self, data_batch, teachers, will_use_aux: bool = False):
        use_batch = transpose(data_batch) if self.will_transpose else data_batch
        converted_teacher = self.__teacher_preprocessor.run_preprpocess(teachers)
        other_batch, other_teachers = build_other_batch(use_batch, converted_teacher)
        main_base_teacher = [labels[0] for labels in converted_teacher]
        main_other_teacher = [labels[0] for labels in other_teachers]
        main_siamese_label = build_siamese_labels_for_space(main_base_teacher, main_other_teacher, self.margin)
        aux_base_teacher = [labels[1] for labels in converted_teacher]
        aux_other_teacher = [labels[1] for labels in other_teachers]
        aux_siamese_label = build_siamese_labels_for_space(aux_base_teacher, aux_other_teacher, self.aux_margin)
        batch_pair = [np.array(use_batch), np.array(other_batch)]
        siamese_label_pair = [main_siamese_label, aux_siamese_label]
        return np.array(batch_pair), np.array(siamese_label_pair, dtype="f4")

    def preprocess_evaluate_original(self, x, y):
        return x, self.__teacher_preprocessor.build_main_teacher(y)

    def preprocess_for_calc_data(self, x, y, is_train=False):
        use_x = transpose(x) if self.will_transpose else x
        if is_train:
            return use_x, self.build_teachers_for_train(y)
        return use_x, self.__teacher_preprocessor.build_main_teacher(y)

    def build_teachers_for_train(self, teachers):
        return self.__teacher_preprocessor.build_teacher_for_train(teachers)
