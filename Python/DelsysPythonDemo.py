import sys
from PySide6.QtWidgets import QApplication
from UIControls.LandingScreenController import *

import argparse

def main(with_classifications_indicator=False, debug=False):
    app = QApplication(sys.argv)
    app.setStyleSheet('.QLabel { font-size: 12pt;}'
                      '.QPushButton { font-size: 12pt;}'
                      '.QListWidget { font-size: 12pt;}'
                      '.QComboBox{ font-size: 12pt;}'
                      )
    controller = LandingScreenController(with_classifications_indicator=with_classifications_indicator,
                                         debug=args.debug)
    sys.exit(app.exec())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--with_classifications_indicator', dest='with_classifications_indicator', action='store_true')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(with_classifications_indicator=False, debug=False)
    args = parser.parse_args()
    
    main(with_classifications_indicator=args.with_classifications_indicator, debug=args.debug)