import sys
from learn_conf.params import ParamBuilder
from keras.preprocessing.image import ImageDataGenerator
from network_model import model_builder as mb
import os
from network_model.learner import split_learn as sl
from keras.callbacks import TensorBoard

cmd_params = sys.argv
conf_path = cmd_params[1]
conf_builder = ParamBuilder.build_from_yaml(conf_path)
path_params = conf_builder.build_path_params()
batch_params = conf_builder.build_batch_params()

IMG_DIR = path_params.dataset_dir
RESULT_DIR = path_params.result_dir
IMG_COLOR = "RGB"
IMG_SIZE = 224
channel = 3 if IMG_COLOR == "RGB" else 1
MODEL_NAME = path_params.model_name
MODEL_RESULT_NAME = path_params.model_result_name
TENSORBOARD_LOG_DIR = './logs'
kernel_size = (3, 3)
fold_num = 5
epoch_num = batch_params.epoch_num
generator_batch_size = batch_params.batch_size
build_original_data_num = 2
preprocess_for_model = None

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    channel_shift_range=1,
    brightness_range=[0.5, 1.0],
    horizontal_flip=True,
    vertical_flip=True)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

TEMP_MODEL_PATH = None
model_generator = mb.build_wrapper(IMG_SIZE,
                                   channel,
                                   MODEL_NAME
                                   )

class_list = os.listdir(os.path.join(IMG_DIR, "train"))
class_list.sort()
print(class_list)
callbacks = [TensorBoard(TENSORBOARD_LOG_DIR)]
model_learner = sl.ModelLearner(model_generator,
                                datagen,
                                test_datagen,
                                class_list,
                                callbacks=callbacks,
                                will_save_h5=True)

built_model = model_learner.train_with_validation(IMG_DIR,
                                                  os.path.join(os.getcwd(), RESULT_DIR),
                                                  epoch_num=epoch_num,
                                                  batch_size=generator_batch_size,
                                                  result_name=MODEL_RESULT_NAME,
                                                  model_name=MODEL_RESULT_NAME,
                                                  tmp_model_path=TEMP_MODEL_PATH,
                                                  monitor='val_acc',
                                                  save_weights_only=False)
