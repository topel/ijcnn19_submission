import os
from random import shuffle

import numpy as np
import tqdm
import copy
from keras.models import Model
from keras.utils import Sequence

class DCASE2018:
    CLIP_LENGTH = 10
    NB_CLASS = 10
    class_correspondance = {"Alarm_bell_ringing": 0, "Speech": 1, "Dog": 2, "Cat": 3, "Vacuum_cleaner": 4,
        "Dishes": 5, "Frying": 6, "Electric_shaver_toothbrush": 7, "Blender": 8, "Running_water": 9}

    class_correspondance_reverse = dict()
    for k in class_correspondance:
        class_correspondance_reverse[class_correspondance[k]] = k

    def __init__(self,
                 featureRoot: str, metaRoot: str, features: list,
                 expandWithUod: bool = False, expandPercent: float = 0.20,
                 validationPercent: float = 0.2,
                 normalizer = None):

        # directories
        self.featureRoot = featureRoot
        self.feat_train_weak = os.path.join(featureRoot, "train", "weak")
        self.feat_train_uid = os.path.join(featureRoot, "train", "unlabel_in_domain")
        self.feat_train_uod = os.path.join(featureRoot, "train", "unlabel_out_of_domain")
        self.feat_test = os.path.join(featureRoot, "test")

        # metadata
        self.metaRoot = metaRoot
        self.meta_train_weak = os.path.join(metaRoot, "weak.csv")
        self.meta_train_uid = os.path.join(metaRoot, "unlabel_in_domain.csv")
        self.meta_train_uod = os.path.join(metaRoot, "unlabel_out_of_domain.csv")
        self.meta_test = os.path.join(metaRoot, "test.csv")

        # dataset parameters
        self.features = features
        self.metadata = {}

        self.expandWithUod = expandWithUod
        self.expandPercent = expandPercent
        self.validationPercent = validationPercent
        self.originalShape = {}
        self.normalizer = normalizer

        self.nbClass = 10
        DCASE2018.NB_CLASS = 10
        if expandWithUod:
            self.nbClass = 11
            DCASE2018.NB_CLASS = 11

        # dataset that will be used
        self.trainingDataset = {}
        self.trainingUidDataset = {}
        self.validationDataset = {}
        self.testingDataset = {}
        self.testFileList = []

        # interior variables
        self.build()


    def __call__(self):
        self.build()

    def build(self):
        self.__init()
        self.__loadMeta()
        self.__expand()
        training_data, validation_data = self.__balancedSplit()

        for f in self.features:
            self.__createDataset(f, training_data, validation_data)
            self.__createTestDataset(f)
            self.__preProcessing(f)

    def __init(self):
        # init dict
        for f in self.features:
            self.trainingDataset[f] = {"input": [], "output": []}
            self.validationDataset[f] = {"input": [], "output": []}
            self.testingDataset[f] = {"input": [], "output": []}
            self.trainingUidDataset[f] = {"input": [], "output": []}
            self.originalShape[f] = None

    def __preProcessing(self, feature: str):
        # save original shape
        print(feature)
        self.originalShape[feature] = self.trainingDataset[feature]["input"][0].shape

        # convert to np.array
        self.trainingDataset[feature]["input"] = np.array(self.trainingDataset[feature]["input"])
        self.validationDataset[feature]["input"] = np.array(self.validationDataset[feature]["input"])
        self.trainingDataset[feature]["output"] = np.array(self.trainingDataset[feature]["output"])
        self.validationDataset[feature]["output"] = np.array(self.validationDataset[feature]["output"])
        self.testingDataset[feature]["input"] = np.array(self.testingDataset[feature]["input"])
        self.testingDataset[feature]["output"] = np.array(self.testingDataset[feature]["output"])


        # normalization
        if self.normalizer is not None:
            print("==== Normalization stage ====")
            self.trainingDataset[feature]["input"] = self.normalizer.fit_transform(self.trainingDataset[feature]["input"])
            self.validationDataset[feature]["input"] = self.normalizer.fit_transform(self.validationDataset[feature]["input"])
            self.testingDataset[feature]["input"] = self.normalizer.fit_transform(self.testingDataset[feature]["input"])

        # extend dataset to have enough dim for conv2D
        self.trainingDataset[feature]["input"] = np.expand_dims(self.trainingDataset[feature]["input"], axis=-1)
        self.validationDataset[feature]["input"] = np.expand_dims(self.validationDataset[feature]["input"], axis=-1)
        self.testingDataset[feature]["input"] = np.expand_dims(self.testingDataset[feature]["input"], axis=-1)

    def __loadMeta(self):
        """ Load the metadata for all subset of the DCASE2018 task4 dataset"""

        def load(meta_dir: str, nbFile: int = None):
            if meta_dir != "":
                with open(meta_dir) as f:
                    data = f.readlines()

                if nbFile is None:
                    return [d.split("\t") for d in data[1:]]
                else:
                    return [d.split("\t") for d in data[1:]][:nbFile]

        # load meta data only on the first features (to keep the order)
        self.metadata["weak"] = load(self.meta_train_weak)
        self.metadata["uid"] = load(self.meta_train_uid)
        self.metadata["test"] = load(self.meta_test)

        # Use to extend training dataset, gather only 0.2 * len(training_dataset) of the uod
        if self.expandWithUod:
            nbFileForUod = len(self.metadata["weak"]) * self.expandPercent
            self.metadata["uod"] = load(self.meta_train_uod, int(nbFileForUod))

    def __expand(self):
        for f in self.metadata["weak"]:
            f[0] = [self.featureRoot, "train", "weak", "feature", f[0]]

        if self.expandWithUod:
            for f in self.metadata["uod"]:
                f[0] = [self.featureRoot, "train", "unlabel_out_of_domain", "feature", f[0]]
            self.metadata["weak"].extend(self.metadata["uod"])

    def loadUID(self) -> dict:
        """ Load the features for the "unlabel_in_domain" dataset.

        It is not done when building the dataset since this part is not always necessarily.

        :return: dict containing the data of the features (the key is the name of the feature)
        """
        # prepare path
        print("meta UID: ", len(self.metadata["uid"]))
        for f in self.metadata["uid"]:
            f[0] = [self.featureRoot, "train", "unlabel_in_domain", "feature", f[0][:-1]]

        # ---- load the features ----
        inputs = {}
        with tqdm.tqdm(total=len(self.metadata["uid"]) * len(self.features), unit="Files") as progress:
            for feature in self.features:
                inputs[feature] = []

                for i in range(len(self.metadata["uid"])):
                    info = self.metadata["uid"][i]
                    pathList = info[0]
                    pathList[3] = feature
                    path = os.path.join(*pathList) + ".npy"

                    if os.path.isfile(path):
                        feat = np.load(path)

                        # preprocessing and add
                        feat = np.expand_dims(feat, axis=-1)
                        inputs[feature].append(feat)

                    progress.update()

                inputs[feature] = np.array(inputs[feature])

        return inputs

    def expandWithUID(self, features: np.array, prediction: list):
        for feature in features:
            self.trainingUidDataset[feature]["input"] = features[feature]
            self.trainingUidDataset[feature]["output"] = np.array(prediction)

    def __balancedSplit(self):
        """ Split the weak subset into a balanced weak training and weak validation subsets"""
        splited = [[] for i in range(self.nbClass)]

        # separate the dataset into the 11 classes
        for info in self.metadata["weak"]:
            if len(info) > 1:
                for cls in info[1].split(","):
                    splited[DCASE2018.class_correspondance[cls.rstrip()]].append(info)
            else:
                splited[DCASE2018.class_correspondance["blank"]].append(info)

        # for each class, split into two (80%, 20%) for training and validation
        training = []
        validation = []
        for cls in splited:
            cutIndex = int(len(cls) * self.validationPercent)
            training.extend(cls[cutIndex:])
            validation.extend(cls[:cutIndex])

        # shuffle and load the features in memory
        shuffle(training)
        shuffle(validation)

        return training, validation

    def __createDataset(self, feature: str, training_data: list, validation_data: list):

        def loadFeatures(subset: dict, toLoad: list):
            subset[feature]["input"] = []

            for info in toLoad:
                pathList = info[0]
                pathList[3] = feature
                path = os.path.join(*pathList) + ".npy"

                if os.path.isfile(path):
                    output = [0] * self.nbClass
                    feat = np.load(path)

                    subset[feature]["input"].append(feat)

                    if len(info) > 1:
                        for cls in info[1].split(","):
                            output[DCASE2018.class_correspondance[cls.rstrip()]] = 1
                    else:
                        output[DCASE2018.class_correspondance["blank"]] = 1

                    subset[feature]["output"].append(output)

        loadFeatures(self.trainingDataset, training_data)
        loadFeatures(self.validationDataset, validation_data)

    def __createTestDataset(self, feature):
        self.testingDataset[feature]["input"] = []

        self.testFileList = os.listdir(os.path.join(self.feat_test, "mel"))

        for file in self.testFileList:
            path = os.path.join(self.feat_test, feature, file)
            f = np.load(path)

            self.testingDataset[feature]["input"].append(f)

    def getInputShape(self, feature):
        shape = self.trainingDataset[feature]["input"][0].shape
        return (shape[0], shape[1], 1)

    def __str__(self):
        output = "-" * 30 + "\n"
        for f in self.features:
            output += "Features: " + f + " --------\n\n"

            output += "Using feature: " + os.path.basename(self.feat_train_weak) + "\n"
            if self.meta_train_uod != "":
                output += "Dataset has been augmented using the unlabel out of domain for \"blank\" class \n"
                output += "%s files added\n\n" % len(self.metadata["uod"])

            output += "Training files: %s\nValidation files: %s\n" % (len(self.trainingDataset[f]["input"]), len(self.validationDataset[f]["input"]))
            output += "Validation ratio: %s" % self.validationPercent

        return output

if __name__=='__main__':
    dataset = DCASE2018(
        featureRoot="/baie/corpus/DCASE2018/task4/FEATURES",
        metaRoot="/baie/corpus/DCASE2018/task4/metadata",
        features=["mel"],
        validationPercent=0.2,
        normalizer=None
    )

    print(dataset.testingDataset["mel"]["input"].shape)
