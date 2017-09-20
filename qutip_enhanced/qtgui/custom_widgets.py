# coding=utf-8
from __future__ import print_function, absolute_import, division

__metaclass__ = type

import numpy as np
import itertools


from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QMenu, QAction
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import pyqtSignal

class QTableWidgetEnhanced(QTableWidget):

    clear_table_contents_signal = pyqtSignal()

    def __init__(self, parent=None):
        super(QTableWidgetEnhanced, self).__init__(parent)
        self.clear_table_contents_signal.connect(self.clear_table_contents_for_signal)

    def column_data(self, column_name):

        out = []
        for row in range(self.rowCount()):
            o = self.item(row, self.column_index(column_name))
            if o.text() != '':
                out.append(o.data(0x0100))
        return out

    @property
    def column_names(self):
        return [self.horizontalHeaderItem(i).text() for i in range(self.columnCount())]

    def column_index(self, column_name):
        return self.column_names.index(column_name)

    def add_rows(self, n_new):
        if n_new > 0:
            n_old = self.rowCount()
            self.setRowCount(n_old+n_new)
            for rc in itertools.product(range(n_old, n_old+n_new), range(self.columnCount())):
                new_item = QTableWidgetItem('')
                new_item.setData(0x0100, None)
                new_item.setFlags(Qt.NoItemFlags)
                self.setItem(rc[0], rc[1], new_item)

    def set_columns(self, desired_total_count, header):
        if desired_total_count != len(header):
            raise Exception("Error: {}{}".format(desired_total_count, header))
        self.setColumnCount(desired_total_count)
        self.setHorizontalHeaderLabels(header)
        for rc in itertools.product(range(self.rowCount()), range(desired_total_count)):
            new_item = QTableWidgetItem('')
            new_item.setData(0x0100, None)
            new_item.setFlags(Qt.NoItemFlags)
            self.setItem(rc[0], rc[1], new_item)

    def append_to_column_parameters(self, column_name, parameters):
        cd = self.column_data(column_name)
        new_params = [i for i in parameters if i not in cd]
        delta_n_row = len(parameters) - self.rowCount()
        if delta_n_row > 0:
            self.add_rows(delta_n_row)
        for row_idx, new_param in zip(len(parameters) - len(new_params) + np.arange(0, len(parameters)), new_params):
            self.item(row_idx, self.column_index(column_name)).setText(str(new_param))
            self.item(row_idx, self.column_index(column_name)).setData(0x0100, new_param)
            self.item(row_idx, self.column_index(column_name)).setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

    def set_column_flags(self, column_name, flag):
        for row in range(self.rowCount()):
            self.item(row, self.column_index(column_name)).setFlags(flag)

    def selected_table_items(self, column_name):
        out = []
        for item in self.selectedItems():
            if item.column() == self.column_index(column_name=column_name):
                out.append(item.data(0x0100))
        return out

    def selected_items_unique_column_indices(self):
        return list(set([i.row() for i in self.selectedItems()]))

    def clear_table_contents(self):
        self.clear_table_contents_signal.emit()

    def clear_table_contents_for_signal(self):
        for idx in itertools.product(range(self.rowCount()), range(self.columnCount())):
            self.clearSelection()
            self.item(idx[0], idx[1]).setText('')
            self.item(idx[0], idx[1]).setData(0x0100, None)
            self.item(idx[0], idx[1]).setFlags(Qt.NoItemFlags)

class QTableWidgetEnhancedDrop(QTableWidgetEnhanced):

    def __init__(self, parent=None):
        super(QTableWidgetEnhancedDrop, self).__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    hdf_file_dropped = pyqtSignal(str)

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            links = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            if len(links) != 1:
                print('Warning: {}, {}'.format(links, len(links)))
                event.ignore()
            elif not links[0].endswith(".hdf"):
                print('Warning: {}, {}'.format(links, links[0][:-4]))
                event.ignore()
            else:
                self.hdf_file_dropped.emit(links[0])
        else:
            event.ignore()

class QTableWidgetEnhancedContextMenu(QTableWidgetEnhanced):

    def contextMenuEvent(self, event):
        self.menu = QMenu(self)
        remove_script_action = QAction('Remove row', self)
        self.menu.addAction(remove_script_action)
        self.menu.popup(QCursor.pos())