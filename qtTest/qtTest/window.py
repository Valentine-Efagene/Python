# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_window(object):
    def setupUi(self, window):
        window.setObjectName("window")
        window.resize(800, 600)
        window.setWindowOpacity(1.0)
        window.setAutoFillBackground(False)
        window.setStyleSheet("""
            QMainWindow{
                background-color: rgb(101, 101, 101);
            }""")

        self.centralwidget = QtWidgets.QWidget(window)
        self.centralwidget.setObjectName("centralwidget")
        window.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(window)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuabout = QtWidgets.QMenu(self.menubar)
        self.menuabout.setObjectName("menuabout")
        window.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(window)
        self.statusbar.setObjectName("statusbar")
        window.setStatusBar(self.statusbar)
        self.actionsave = QtWidgets.QAction(window)
        self.actionsave.setObjectName("actionsave")
        self.actionclose = QtWidgets.QAction(window)
        self.actionclose.setObjectName("actionclose")
        self.menuFile.addAction(self.actionsave)
        self.menuFile.addAction(self.actionclose)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuabout.menuAction())

        self.retranslateUi(window)
        QtCore.QMetaObject.connectSlotsByName(window)

    def retranslateUi(self, window):
        _translate = QtCore.QCoreApplication.translate
        window.setWindowTitle(_translate("window", "Test"))
        self.menuFile.setTitle(_translate("window", "File"))
        self.menuabout.setTitle(_translate("window", "&About"))
        self.actionsave.setText(_translate("window", "save"))
        self.actionclose.setText(_translate("window", "close"))

