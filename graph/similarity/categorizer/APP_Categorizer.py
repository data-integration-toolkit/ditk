# -*- coding: ms949 -*-


import wx

import APP_MainWindow

modules = {'MainWindow' : [1, 'Main frame of Application', 'none://APP_MainWindow.py']}

class BoaApp(wx.App):
    def OnInit(self):
        self.main = APP_MainWindow.create(None)
        self.main.Show()
        self.SetTopWindow(self.main)
        return True

def main():
    application = BoaApp(0)
    application.MainLoop()

if __name__ == '__main__':
    main()
