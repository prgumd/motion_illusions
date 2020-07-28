###############################################################################
#
# File: signal_plot.py
#
# A session object that allows logging of values and returning an image
# visualizing the signal
#
# History:
# 06-22-20 - Levi Burner - Created file
#
###############################################################################

import numpy as np
import cv2

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

class SignalPlot(object):
    __instances = {}

    @staticmethod
    def get_instance(session=None):
        try:
            t = SignalPlot.__instances[session]
        except KeyError:
            SignalPlot.__instances[session] = SignalPlot()

        return SignalPlot.__instances[session]

    def __init__(self):
        self._samples = []
        self._sample_shape = None
        self.max_samples = 50

    def add_sample(self, sample):
        if self._sample_shape is None:
            self._sample_shape = sample.shape
        else:
            if self._sample_shape != sample.shape:
                raise ValueError('Sample of different dimension than first cannot be accepted')

        self._samples.append(sample)
        if len(self._samples) > self.max_samples:
        	self._samples.pop(0)

    def plot_signal(self, res, labels):
        fig = Figure(figsize=(res[1]/100.0, res[0]/100.0), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.gca()
        ax.plot(range(len(self._samples)), self._samples)
        ax.set_ylim(0.0, 1.0)
        ax.legend(labels)
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        X = np.fromstring(s, np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(X, cv2.COLOR_RGBA2BGR)
