# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Python\qutip_enhanced\qutip_enhanced/qtgui/plot_data.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_window(object):
    def setupUi(self, window):
        window.setObjectName("window")
        window.resize(1115, 994)
        self.parameter_tab = QtWidgets.QTabWidget(window)
        self.parameter_tab.setGeometry(QtCore.QRect(6, -1, 1101, 291))
        self.parameter_tab.setObjectName("parameter_tab")
        self.plot_tab = QtWidgets.QWidget()
        self.plot_tab.setObjectName("plot_tab")
        self.observation_widget = QtWidgets.QListWidget(self.plot_tab)
        self.observation_widget.setGeometry(QtCore.QRect(770, 0, 331, 261))
        self.observation_widget.setObjectName("observation_widget")
        self.parameter_table = QTableWidgetEnhanced(self.plot_tab)
        self.parameter_table.setGeometry(QtCore.QRect(0, 0, 761, 261))
        self.parameter_table.setObjectName("parameter_table")
        self.parameter_table.setColumnCount(0)
        self.parameter_table.setRowCount(0)
        self.parameter_tab.addTab(self.plot_tab, "")
        self.fit_tab = QtWidgets.QWidget()
        self.fit_tab.setObjectName("fit_tab")
        self.fit_select_table = QTableWidgetEnhanced(self.fit_tab)
        self.fit_select_table.setGeometry(QtCore.QRect(0, 0, 421, 251))
        self.fit_select_table.setObjectName("fit_select_table")
        self.fit_select_table.setColumnCount(0)
        self.fit_select_table.setRowCount(0)
        self.fit_result_table = QTableWidgetEnhanced(self.fit_tab)
        self.fit_result_table.setGeometry(QtCore.QRect(430, 0, 661, 251))
        self.fit_result_table.setObjectName("fit_result_table")
        self.fit_result_table.setColumnCount(0)
        self.fit_result_table.setRowCount(0)
        self.parameter_tab.addTab(self.fit_tab, "")
        self.info = QtWidgets.QPlainTextEdit(window)
        self.info.setGeometry(QtCore.QRect(873, 360, 231, 621))
        self.info.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.info.setReadOnly(True)
        self.info.setBackgroundVisible(False)
        self.info.setObjectName("info")
        self.gridLayoutWidget = QtWidgets.QWidget(window)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 290, 851, 631))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.plot_layout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setObjectName("plot_layout")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(window)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(9, 919, 851, 61))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.toolbar_layout = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.toolbar_layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar_layout.setObjectName("toolbar_layout")
        self.update_plot_button = QtWidgets.QPushButton(window)
        self.update_plot_button.setGeometry(QtCore.QRect(870, 290, 91, 51))
        self.update_plot_button.setObjectName("update_plot_button")
        self.update_fit_result_button = QtWidgets.QPushButton(window)
        self.update_fit_result_button.setGeometry(QtCore.QRect(970, 290, 131, 51))
        self.update_fit_result_button.setObjectName("update_fit_result_button")

        self.retranslateUi(window)
        self.parameter_tab.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(window)

    def retranslateUi(self, window):
        _translate = QtCore.QCoreApplication.translate
        window.setWindowTitle(_translate("window", "Form"))
        self.parameter_tab.setTabText(self.parameter_tab.indexOf(self.plot_tab), _translate("window", "Plot"))
        self.parameter_tab.setTabText(self.parameter_tab.indexOf(self.fit_tab), _translate("window", "Fit"))
        self.update_plot_button.setText(_translate("window", "UpdatePlot"))
        self.update_fit_result_button.setText(_translate("window", "UpdateFit"))

from qutip_enhanced.qtgui.custom_widgets import QTableWidgetEnhanced
