# coding=utf-8
from __future__ import print_function, absolute_import, division

__metaclass__ = type

import numpy as np
import itertools


from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtGui import QTextCharFormat, QSyntaxHighlighter, QColor, QFont
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem


class Highlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super(Highlighter, self).__init__(parent)

        keywordFormat = QTextCharFormat()
        keywordFormat.setForeground(Qt.darkBlue)
        keywordFormat.setFontWeight(QFont.Bold)

        keywordPatterns = ["\\bchar\\b", "\\bclass\\b", "\\bconst\\b",
                           "\\bdouble\\b", "\\benum\\b", "\\bexplicit\\b", "\\bfriend\\b",
                           "\\binline\\b", "\\bint\\b", "\\blong\\b", "\\bnamespace\\b",
                           "\\boperator\\b", "\\bprivate\\b", "\\bprotected\\b",
                           "\\bpublic\\b", "\\bshort\\b", "\\bsignals\\b", "\\bsigned\\b",
                           "\\bslots\\b", "\\bstatic\\b", "\\bstruct\\b",
                           "\\btemplate\\b", "\\btypedef\\b", "\\btypename\\b",
                           "\\bunion\\b", "\\bunsigned\\b", "\\bvirtual\\b", "\\bvoid\\b",
                           "\\bvolatile\\b"]

        self.highlightingRules = [(QRegExp(pattern), keywordFormat)
                                  for pattern in keywordPatterns]

        classFormat = QTextCharFormat()
        classFormat.setFontWeight(QFont.Bold)
        classFormat.setForeground(Qt.darkMagenta)
        self.highlightingRules.append((QRegExp("\\bQ[A-Za-z]+\\b"),
                                       classFormat))

        singleLineCommentFormat = QTextCharFormat()
        singleLineCommentFormat.setForeground(Qt.red)
        self.highlightingRules.append((QRegExp("//[^\n]*"),
                                       singleLineCommentFormat))

        self.multiLineCommentFormat = QTextCharFormat()
        self.multiLineCommentFormat.setForeground(Qt.red)

        quotationFormat = QTextCharFormat()
        quotationFormat.setForeground(Qt.darkGreen)
        self.highlightingRules.append((QRegExp("\".*\""),
                                       quotationFormat))

        functionFormat = QTextCharFormat()
        functionFormat.setFontItalic(True)
        functionFormat.setForeground(Qt.blue)
        self.highlightingRules.append((QRegExp("\\b[A-Za-z0-9_]+(?=\\()"),
                                       functionFormat))

        self.commentStartExpression = QRegExp("/\\*")
        self.commentEndExpression = QRegExp("\\*/")

    def highlightBlock(self, text):
        for pattern, format in self.highlightingRules:
            expression = QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)

        self.setCurrentBlockState(0)

        startIndex = 0
        if self.previousBlockState() != 1:
            startIndex = self.commentStartExpression.indexIn(text)

        while startIndex >= 0:
            endIndex = self.commentEndExpression.indexIn(text, startIndex)

            if endIndex == -1:
                self.setCurrentBlockState(1)
                commentLength = len(text) - startIndex
            else:
                commentLength = endIndex - startIndex + self.commentEndExpression.matchedLength()

            self.setFormat(startIndex, commentLength,
                           self.multiLineCommentFormat)
            startIndex = self.commentStartExpression.indexIn(text, startIndex + commentLength);


def format(color, style=''):
    """Return a QTextCharFormat with the given attributes.
    """
    _color = QColor()
    _color.setNamedColor(color)

    _format = QTextCharFormat()
    _format.setForeground(_color)
    if 'bold' in style:
        _format.setFontWeight(QFont.Bold)
    if 'italic' in style:
        _format.setFontItalic(True)

    return _format


# Syntax styles that can be shared by all languages
STYLES = {
    'import': format('blue', 'italic'),
    'plaintext': format('black'),
    'keyword': format('blue'),
    'operator': format('red', 'bold'),
    'brace': format('darkGray'),
    'defclass': format('black', 'bold'),
    'string': format('magenta'),
    'string2': format('darkMagenta'),
    'comment': format('darkGreen', 'italic'),
    'self': format('green', 'italic'),
    'numbers': format('brown'),
}


class PythonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for the Python language.
    """
    # Python keywords
    keywords = [
        'and', 'assert', 'break', 'class', 'continue', 'def',
        'del', 'elif', 'else', 'except', 'exec', 'finally',
        'for', 'from', 'global', 'if', 'import', 'in',
        'is', 'lambda', 'not', 'or', 'pass', 'print',
        'raise', 'return', 'try', 'while', 'yield',
        'None', 'True', 'False',
    ]

    # Python operators
    operators = [
        '=',
        # Comparison
        '==', '!=', '<', '<=', '>', '>=',
        # Arithmetic
        '\+', '-', '\*', '/', '//', '\%', '\*\*',
        # In-place
        '\+=', '-=', '\*=', '/=', '\%=',
        # Bitwise
        '\^', '\|', '\&', '\~', '>>', '<<',
    ]

    # Python braces
    braces = [
        '\{', '\}', '\(', '\)', '\[', '\]',
    ]

    def __init__(self, document):
        QSyntaxHighlighter.__init__(self, document)

        # Multi-line strings (expression, flag, style)
        # FIXME: The triple-quotes in these two lines will mess up the
        # syntax highlighting from this point onward
        self.tri_single = (QRegExp("'''"), 1, STYLES['string2'])
        self.tri_double = (QRegExp('"""'), 2, STYLES['string2'])

        rules = []

        # Keyword, operator, and brace rules
        rules += [(r'\b%s\b' % w, 0, STYLES['keyword'])
                  for w in PythonHighlighter.keywords]
        rules += [(r'%s' % o, 0, STYLES['operator'])
                  for o in PythonHighlighter.operators]
        rules += [(r'%s' % b, 0, STYLES['brace'])
                  for b in PythonHighlighter.braces]

        # All other rules
        rules += [
            (r'import', 0, STYLES['import']),
            # (r'.', 0, STYLES['plaintext']),
            # 'self'
            (r'\bself\b', 0, STYLES['self']),

            # Double-quoted string, possibly containing escape sequences
            (r'"[^"\\]*(\\.[^"\\]*)*"', 0, STYLES['string']),
            # Single-quoted string, possibly containing escape sequences
            (r"'[^'\\]*(\\.[^'\\]*)*'", 0, STYLES['string']),

            # 'def' followed by an identifier
            (r'\bdef\b\s*(\w+)', 1, STYLES['defclass']),
            # 'class' followed by an identifier
            (r'\bclass\b\s*(\w+)', 1, STYLES['defclass']),

            # From '#' until a newline
            (r'#[^\n]*', 0, STYLES['comment']),

            # Numeric literals
            (r'\b[+-]?[0-9]+[lL]?\b', 0, STYLES['numbers']),
            (r'\b[+-]?0[xX][0-9A-Fa-f]+[lL]?\b', 0, STYLES['numbers']),
            (r'\b[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\b', 0, STYLES['numbers']),
        ]

        # Build a QRegExp for each pattern
        self.rules = [(QRegExp(pat), index, fmt)
                      for (pat, index, fmt) in rules]

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text.
        """
        # Do other syntax formatting
        for expression, nth, format in self.rules:
            index = expression.indexIn(text, 0)

            while index >= 0:
                # We actually want the index of the nth match
                index = expression.pos(nth)
                length = len(expression.cap(nth))
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)

        self.setCurrentBlockState(0)

        # Do multi-line strings
        in_multiline = self.match_multiline(text, *self.tri_single)
        if not in_multiline:
            in_multiline = self.match_multiline(text, *self.tri_double)

    def match_multiline(self, text, delimiter, in_state, style):
        """Do highlighting of multi-line strings. ``delimiter`` should be a
        ``QRegExp`` for triple-single-quotes or triple-double-quotes, and
        ``in_state`` should be a unique integer to represent the corresponding
        state changes when inside those strings. Returns True if we're still
        inside a multi-line string when this function is finished.
        """
        # If inside triple-single quotes, start at 0
        if self.previousBlockState() == in_state:
            start = 0
            add = 0
        # Otherwise, look for the delimiter on this line
        else:
            start = delimiter.indexIn(text)
            # Move past this match
            add = delimiter.matchedLength()

        # As long as there's a delimiter match on this line...
        while start >= 0:
            # Look for the ending delimiter
            end = delimiter.indexIn(text, start + add)
            # Ending delimiter on this line?
            if end >= add:
                length = end - start + add + delimiter.matchedLength()
                self.setCurrentBlockState(0)
            # No; multi-line string
            else:
                self.setCurrentBlockState(in_state)
                length = text.length() - start + add
            # Apply formatting
            self.setFormat(start, length, style)
            # Look for the next match
            start = delimiter.indexIn(text, start + length)

        # Return True if still inside a multi-line string, False otherwise
        if self.currentBlockState() == in_state:
            return True
        else:
            return False

class QTableWidgetEnhanced(QTableWidget):

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
        for idx in itertools.product(range(self.rowCount()), range(self.columnCount())):
            self.clearSelection()
            self.item(idx[0], idx[1]).setText('')
            self.item(idx[0], idx[1]).setData(0x0100, None)
            self.item(idx[0], idx[1]).setFlags(Qt.NoItemFlags)