from DataCollector.CollectDataWindow import CollectDataWindow
from StartMenu.StartWindow import StartWindow


class LandingScreenController():
    def __init__(self, with_classifications_indicator=False, debug=False, no_classification=False):
        self.startWindow = StartWindow(self)
        self.collectWindow = CollectDataWindow(self, with_classifications_indicator=with_classifications_indicator)

        self.startWindow.show()

        self.curHeight = 900
        self.curWidth = 1400
        
        self.debug = debug
        self.no_classification = no_classification

    def showStartMenu(self):
        self.collectWindow.close()
        self.startWindow.show()

    def showCollectData(self):
        self.startWindow.close()
        if self.startWindow.plot_enabled.isChecked():
            self.collectWindow.plot_enabled = True
            self.collectWindow.AddPlotPanel()
        self.collectWindow.SetCallbackConnector(debug=self.debug, no_classification=self.no_classification)
        self.collectWindow.connect_callback()
        self.collectWindow.show()
