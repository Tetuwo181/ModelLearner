import yaml


class PathParams(object):
    def __init__(self, raw_params):
        self.__params = raw_params

    @property
    def dataset_dir(self):
        return self.__params["img_dir"]

    @property
    def result_dir(self):
        return self.__params["result_dir"]

    @property
    def model_name(self):
        return self.__params["model_name"]

    @property
    def model_result_name(self):
        return self.__params["model_result_name"]


class BatchParams(object):
    def __init__(self, raw_params):
        self.__params = raw_params

    @property
    def batch_size(self):
        return self.__params["batch_size"]

    @property
    def epoch_num(self):
        return self.__params["epoch_num"]

    @property
    def bagginng_num(self):
        return self.__params["bagginng_num"]

    @property
    def bagging_choice_rate(self):
        return self.__params["bagging_choice_rate"]


class ParamBuilder(object):

    @staticmethod
    def build_from_yaml(yaml_path):
        with open(yaml_path, 'r') as yml:
            config = yaml.load(yml)
            print(config)
            return ParamBuilder(config)

    def __init__(self, raw_params):
        self.__params = raw_params

    def build_path_params(self):
        return PathParams(self.__params["path"])

    def build_batch_params(self):
        return BatchParams(self.__params["batch"])
