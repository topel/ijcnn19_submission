import sys, os
import numpy as np
import copy

from Binarizer import Binarizer
from datasetGenerator import DCASE2018

class Encoder:
    def __init__(self):
        self.frameLength = 0
        self.nbFrame = 0

    def __pad(self, array: list, window_size: int, method: str = "same"):
        """ Pad and array using the methods given and a window_size.

        :param array: the array to pad
        :param window_size: the size of the working window
        :param method: methods of padding, two available "same" | "reflect"
        :return: the padded array
        """
        missing = len(array) % window_size
        output = copy.copy(array)

        if missing > 0:
            output = np.concatenate((output, [output[-1]] * missing))

        return output

    def __smooth(self, temporalPrediction: np.array, method: str = "smoothMovingAvg", **kwargs) -> np.array:
        _methods = ["smoothMovingAvg"]
        if method not in _methods:
            print("method %s doesn't exist. Only ", _methods, " available")
            sys.exit(1)

        if method == _methods[0]: smoother = self.__smoothMeanAvg
        else:
            return

        return smoother(temporalPrediction, **kwargs)

    def __smoothMeanAvg(self, temporalPrediction: np.array, **kwargs):
        print("Smooting using the smooth moving average algorithm")
        def smooth(data, window_len = 11):
            # retreiving extract arguments
            keys = kwargs.keys()
            window_len = kwargs["window_len"] if "window_len" in keys else 11
            weight = kwargs["weight"] if "weight" in keys else 2

            window_len = int(window_len)

            if window_len < 3:
                return data

            s = np.r_[weight * data[0] - data[window_len - 1::-1], data, weight * data[-1] - data[-1:-window_len:-1]]
            w = np.ones(window_len, 'd')
            y = np.convolve(w / w.sum(), s, mode='same')
            return y[window_len:-window_len + 1]

        outputs = []
        for clipInd in range(len(temporalPrediction)):
            clip = temporalPrediction[clipInd]
            curves = []
            for clsInd in range(len(clip.T)):
                #clip.T[clsInd] = smooth(clip.T[clsInd])
                curves.append(smooth(clip.T[clsInd]))
            curves = np.array(curves)
            outputs.append(curves.T)
        outputs = np.array(outputs)
        return outputs

    def binToClass(self, prediction: np.array, binarize: bool = False) -> np.array:
        """ Given the prediction output of the network, match the results to the class name.

        Usefull in the case of labelisation (for unlabel_in_domain)
        :param prediction: a 2 dimension numpy array, result of the prediction of the model
        :param binarize: If the prediction need to be binarized first.
        :return the prediction list with the class name instaed if the class number
        """
        output = []

        if binarize:
            b = Binarizer()
            prediction = b.binarize(prediction)

        for pred in prediction:
            nbpredict = 0

            label = ""
            for i in range(len(pred)):
                isPredict = pred[i]

                if isPredict == 1:
                    label += "%s," % DCASE2018.class_correspondance_reverse[i]
                    nbpredict += 1

            if nbpredict > 0:
                output.append(label[:-1])
            else:
                output.append(label)

        return np.array(output)

    def encode(self, temporalPrediction: np.array, method: str = "threshold", smooth: str = None, **kwargs) -> str:
        """
        Perform the localization of the sound event present in the file.

        Using the temporal prediction provided y the last step of the system, it will "localize" the sound event
        inside the file under the form of a strongly annotated line. (see DCASE2018 task 4 strong label exemple).
        There is two methods implemented here, one using a simple threshold based segmentation and an other using
        a modulation system based on the variance of the prediction over the time.

        :param temporalPrediction: A 3-dimension numpy array (<nb clip>, <nb frame>, <nb class>)
        :return: The result of the system under the form of a strong annotation text where each represent on timed event
        """
        # parameters verification
        _methods=["threshold", "hysteresis", "derivative", "primitive", "dynamic-threshold"]
        if method not in _methods:
            print("method %s doesn't exist. Only", _methods, " available")
            sys.exit(1)

        if method == _methods[0]: encoder = self.__encodeUsingThreshold
        elif method == _methods[2]: encoder = self.__encodeUsingDerivative
        elif method == _methods[1]: encoder = self.__encodeUsingHysteresis
        elif method == _methods[3]: encoder = self.__encodeUsingPrimitive
        else:
            sys.exit(1)

        if smooth is not None:
            temporalPrediction = self.__smooth(temporalPrediction, method=smooth, **kwargs)

        self.nbFrame = temporalPrediction.shape[1]
        self.frameLength = 10 / self.nbFrame
        return encoder(temporalPrediction, **kwargs)

    def __encodeUsingHysteresis(self, temporalPrediction: np.array, **kwargs) -> list:
        """ Hysteresys based localization of the sound event in the clip using the temporal prediction.

        :param temporalPrediction: A 3-dimension numpy array (<nb clip>, <nb frame>, <nb class>)
        :param low: low thresholds
        :param high: high thresholds
        :return: the result of the system under the form of a strong annotation text where each line represent on timed event
        """
        low = kwargs["low"] if "low" in kwargs.keys() else 0.4
        high = kwargs["high"] if "high" in kwargs.keys() else 0.6
        prediction = temporalPrediction

        output = []

        for clip in prediction:
            labeled = dict()

            cls = 0
            for predictionPerClass in clip.T:
                converted = list()
                segment = [0, 0]
                nbSegment = 1
                for i in range(len(predictionPerClass)):
                    element = predictionPerClass[i]

                    # first element
                    if i == 0:
                        segment = [1.0, 1] if element > high else [0.0, 1]

                    # then
                    if element > high and segment[0] == 1:
                        segment[1] += 1

                    elif element > high and segment[0] == 0:
                        converted.append(segment)
                        nbSegment += 1
                        segment = [1.0, 1]

                    elif element <= low and segment[0] == 0:
                        segment[1] += 1

                    elif element <= low and segment[0] == 1:
                        converted.append(segment)
                        nbSegment += 1
                        segment = [0.0, 0]

#                 if nbSegment == 1:
                converted.append(segment)

                labeled[cls] = copy.copy(converted)
                cls += 1

            output.append(labeled)

        return output

    def __encodeUsingThreshold(self, temporalPrediction: np.array, **kwargs) -> list:
        """ Threshold based localization of the sound event in the clip using the temporal prediction.

        :param temporalPrediction: A 3-dimension numpy array (<nb clip>, <nb frame>, <nb class>)
        :param kwargs: Extra arguments. None possible in this method
        :return: The result of the system under the form of a strong annotation text where each represent on timed event
        """
        output = []
        temporalPrecision = 200        # ms

        # binarize the results using the thresholds (default or optimized) provided by the Binarizer
        binarizer = Binarizer()
        binPrediction = binarizer.binarize(temporalPrediction)

        # Merging "hole" that are smaller than 200 ms
        stepLength = DCASE2018.CLIP_LENGTH / temporalPrediction.shape[1] * 1000     # in ms
        maxHoleSize = int(temporalPrecision / stepLength)

        for clip in binPrediction:
            labeled = dict()

            cls = 0
            for binPredictionPerClass in clip.T:
                # convert the binarized list into a list of tuple representing the element and it's number of
                # occurrence. The order is conserved and the total sum should be equal to 10s

                # first pass --> Fill the holes
                for i in range(len(binPredictionPerClass) - maxHoleSize):
                    window = binPredictionPerClass[i : i+maxHoleSize]

                    if window[0] == window[-1] == 1:
                        window[:] = [window[0]] * maxHoleSize

                # second pass --> split into segments
                converted = []
                cpt = 0
                nbSegment = 0
                previousElt = None
                for element in binPredictionPerClass:
                    if previousElt is None:
                        previousElt = element
                        cpt += 1
                        nbSegment = 1
                        continue

                    if element == previousElt:
                        cpt += 1

                    else:
                        converted.append((previousElt, cpt))
                        previousElt = element
                        nbSegment += 1
                        cpt = 1

                # case where the class is detect during the whole clip
                if nbSegment == 1:
                    converted.append((previousElt, cpt))

                labeled[cls] = copy.copy(converted)
                cls += 1

            output.append(labeled)

        return output

    def __encodeUsingDerivative(self, temporalPrediction: np.array, **kwargs) -> list:
        """ Threshold based localization of the sound event in the clip using the temporal prediction.

        :param temporalPrediction: A 3-dimension numpy array (<nb clip>, <nb frame>, <nb class>)
        :return: The result of the system under the form of a strong annotation text where each represent on timed event
        """

        def futureIsFlat(prediction: np.array, currentPos: int, flat: float = 0.05, window_size: int = 5) -> bool:
            """
            Detect what is following is "kinda" flat.
            :param prediction: The prediction values of the current class
            :param currentPos: The current position of the window (left side)
            :return: True is the near future of the curve is flat, False otherwise
            """
            slopes = 0

            # if not future possible (end of the curve)
            if (currentPos + 2 * window_size) > len(prediction):
                return False

            # sum the slope value for the next <window_size> window
            for i in range(currentPos, currentPos + 2 * window_size):
                window = prediction[i:i + window_size]
                slopes += window[-1] - window[0]

            averageSlope = slopes / 2 * window_size

            # is approximately flat, the return True, else False
            return abs(averageSlope) < flat

        # retreive the argument from kwargs
        keys = kwargs.keys()
        rising = kwargs["rising"] if "rising" in keys else 0.5
        decreasing = kwargs["decreasing"] if "decreasing" in keys else -0.5
        flat = kwargs["flat"] if "flat" in keys else 0.05
        window_size = kwargs["window_size"] if "window_size" in keys else 5
        high = kwargs["high"] if "high" in keys else 0.5

        output = []

        for clip in temporalPrediction:
            cls = 0
            labeled = dict()

            for predictionPerClass in clip.T:

                nbSegment = 1
                segments = []
                segment = [0.0, 0]
                for i in range(len(predictionPerClass) - window_size):
                    window = predictionPerClass[i:i+window_size]

                    slope = window[-1] - window[0]

                    # first element
                    if i == 0:
                        segment = [1.0, 1] if window[0] > high else [0.0, 1]

                    # rising slope while on "low" segment --> changing segment
                    if slope > rising and segment[0] == 0:
                        segments.append(segment)
                        nbSegment += 1
                        segment = [1.0, 1]

                    # rising slope while on "high" segment --> same segment
                    elif slope > rising and segment[0] == 1:
                        segment[1] += 1

                    # decreasing slope while on "low" segment --> same segment
                    elif slope < decreasing and segment[0] == 0:
                        segment[1] += 1

                    # decreasing slope while on "high" segment --> one extra condition, future is flat ?
                    elif slope < decreasing and segment[0] == 1:
                        # is there is no flat plateau right after --> same segment
                        if not futureIsFlat(predictionPerClass, i, flat, window_size):
                            segment[1] += 1

                        # Otherwise --> change segment
                        else:
                            segments.append(segment)
                            nbSegment += 1
                            segment = [0.0, 1]


                    else:
                        segment[1] += 1

                if nbSegment == 1:
                    segments.append(copy.copy(segment))

                labeled[cls] = segments
                cls += 1

            output.append(labeled)
        return output

    def __encodeUsingPrimitive(self, temporalPrediction: np.array, **kwargs) -> list:
        """ Area under the curve based localization of the sound event using the temporal prediction.

        Given a sliding window, the area under the curve of the window is computed and, The area under the curve
        is computed and depending on whether it is above a threshold or not the segment will be considered.
        implementation based of the composite trapezoidal rule.

        :param temporalPrediction: A 3-dimension numpy array (<nb clip>, <nb frame>, <nb class>)
        :param kwargs: Extra arguments like "window_size" and "threshold"
        :return: The result of the system under the form of a strong annotation text where each represent on timed event
        """

        def area(window: list) -> float:
            """ Compute the area under the curve inside a window

            :param window: the current window
            :return: the area under the curve
            """
            area = 0
            for i in range(len(window) - 1):
                area += (window[i+1] + window[i]) / 2

            return area

        # retreiving extra arguments
        keys = kwargs.keys()
        window_size = kwargs["window_size"] if "window_size" in keys else 5
        threshold = kwargs["threshold"] if "threshold" in keys else window_size / 4
        stride = kwargs["stride"] if "stride" in keys else 1
        padding = kwargs["padding"] if "padding" in keys else "same"

        output = []
        for clip in temporalPrediction:
            labeled = dict()
            cls = 0
            for predictionPerClass in clip.T:
                paddedPredictionPerClass = self.__pad(predictionPerClass, window_size, method=padding)

                nbSegment = 1
                segments = []
                segment = None
                for i in range(0, len(paddedPredictionPerClass) - window_size, stride):
                    window = paddedPredictionPerClass[i:i+window_size]
                    wArea = area(window)

                    # first element
                    if i == 0:
                        segment = [1.0, 1] if wArea > threshold else [0.0, 1]

                    # then
                    if wArea > threshold and segment[0] == 1:
                        segment[1] += 1

                    elif wArea > threshold and segment[0] == 0:
                        segments.append(segment)
                        nbSegment += 1
                        segment = [1.0, 1]

                    elif wArea <= threshold and segment[0] == 0:
                        segment[1] += 1

                    elif wArea <= threshold and segment[0] == 1:
                        segments.append(segment)
                        nbSegment += 1
                        segment = [0.0, 1]

                if nbSegment == 1:
                    segments.append(segment)

                labeled[cls] = copy.copy(segments)
                cls += 1

            output.append(labeled)
        return output

    def parse(self, allSegments: list, testFilesName: list) -> str:
        """ Transform a list of segment into a txt file ready for evaluation.

        :param allSegments: a list of dict of 10 key. the list length is equal to the number of file, the dict number
        of key to the number of class
        :param testFilesName: the list of filename in the same order than the list allSegments
        :return: a str file ready for evaluation using dcase_util evaluation_measure.py
        """
        output = ""

        for clipIndex in range(len(allSegments)):
            clip = allSegments[clipIndex]

            for cls in clip:
                start = 0

                for segment in clip[cls]:
                    if segment[0] == 1.0:
                        output += "%s\t%f\t%f\t%s\n" % (
                            os.path.basename(testFilesName[clipIndex])[:-4],
                            start * self.frameLength,
                            (start + segment[1]) * self.frameLength,
                            DCASE2018.class_correspondance_reverse[cls]
                        )
                    start += segment[1]

        return output

if __name__=='__main__':
    import random
    e = Encoder()

    # create fake data (temporal prediction)
    def mRandom():
        r = random.random()
        return r

    def fakeTemporalPrediction():
        prediction = []
        for i in range(10):
            clip = []
            for j in range(200):
                score = [mRandom() for k in range(10)]
                clip.append(score)
            prediction.append(clip)

        prediction = np.array(prediction)

        #o = e.encode(prediction)       # basic thresold with hold filling
        o = e.encode(prediction, method="primitive")
        for k in o:
            print(len(k[0]), k[0])
        t = e.parse(o, prediction[:,0,0])


    fakeTemporalPrediction()
