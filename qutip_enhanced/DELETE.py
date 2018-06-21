from __future__ import division

import seaborn as sns
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QListWidgetItem
import PyQt5.QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib import pyplot as plt

class MatplotWidget1(FigureCanvas):

    def __init__(self, parent=None, dpi=100, initial_message=None):
        fig,ax = plt.subplots()
        #fig = Figure(figsize=(5, 5), dpi=dpi, tight_layout=True)
        #self.fig = plt.figure()
        #self.axes = plt.Axes()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, PyQt5.QtWidgets.QSizePolicy.Expanding, PyQt5.QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()
        palette = self.palette()
        self.setContentsMargins(0,0,0,0)
        # fig.set_facecolor(palette.Background().color().getRgbF()[0:3])
        self.axes = ax


    def make_plot(self, data,x_name,z_name,outcome):
        sns.set_style("darkgrid")

        plt.sca(self.axes)
        plt.clf()
        pg=sns.lmplot(x_name,outcome,data,hue=z_name)

        #ax = plt.gca()
        #ax.legend(numpoints=1, fancybox=True, fontsize="small", )
        #self.axes.get_legend().draggable(True, update="loc")
        fig = pg.fig
        fig.set_canvas(self)
        self.figure = fig
        fig = self.figure
        palette = self.palette()
        # fig.set_facecolor(palette.background().color().getRgbF()[0:3])

        plt.show()
        self.draw()
        self.resize_event()
        self.draw()


if __name__ == "__main__":

    widget = MatplotWidget1()
    widget.show()
    #create fake data
    n_subjects = 40
    d = {
        "Group1" : np.random.randint(1,4,n_subjects),
        "Group2" : np.random.randint(1,3,n_subjects),
        "Outcome": np.random.random(n_subjects)
    }
    data = pd.DataFrame(d)
    widget.make_plot(data,"Group1","Group2","Outcome")
    app = PyQt5.QtWidgets.QApplication([])
app.exec_()