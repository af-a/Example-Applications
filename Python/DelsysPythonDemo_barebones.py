
import os
import json
import socket
import time
import argparse
import threading

from collections import deque

# Note: only tested on Windows:
import keyboard
import pandas as pd

# from franka_emg_grasping.emg_classification import EMGClassifier
from franka_emg_grasping.emg_classification_refactored import EMGClassifier

from DataCollector.ClassificationIndicatorWindow import ClassificationsWindow

## Delsys API imports:
from AeroPy.DataManager import *
from Export.CsvWriter import CsvWriter
from pythonnet import load; load("coreclr"); import clr
clr.AddReference("System.Collections")
clr.AddReference("resources/DelsysAPI")
clr.AddReference("System.Collections")

from Aero import AeroPy

key = "MIIBKjCB4wYHKoZIzj0CATCB1wIBATAsBgcqhkjOPQEBAiEA/////wAAAAEAAAAAAAAAAAAAAAD///////////////8wWwQg/////wAAAAEAAAAAAAAAAAAAAAD///////////////wEIFrGNdiqOpPns+u9VXaYhrxlHQawzFOw9jvOPD4n0mBLAxUAxJ02CIbnBJNqZnjhE50mt4GffpAEIQNrF9Hy4SxCR/i85uVjpEDydwN9gS3rM6D0oTlF2JjClgIhAP////8AAAAA//////////+85vqtpxeehPO5ysL8YyVRAgEBA0IABAEu4d8Qg556AwxaLhAUNtKilChZytPqKQ/I+F/cx/hOIr7SzVtZbHqiI6eVOHtHInBFU+suljbYB0wtvmSts7E="
license = "<License>  <Id>40ffa06a-3940-4d03-ae7a-4d0eb5d9b482</Id>  <Type>Standard</Type>  <Quantity>10</Quantity>  <LicenseAttributes>    <Attribute name='Software'></Attribute>  </LicenseAttributes>  <ProductFeatures>    <Feature name='Sales'>True</Feature>    <Feature name='Billing'>False</Feature>  </ProductFeatures>  <Customer>    <Name>Technical University Munich</Name>    <Email>mhilman.fatoni@tum.de</Email>  </Customer>  <Expiration>Mon, 25 Sep 2034 23:00:00 GMT</Expiration>  <Signature>MEUCIHJ0XgUAq3Zhv4MOMqRP9pS4PDfuK2+ZZHJ4hELQ6sN/AiEAgsE1wCys+N6jHxkYErNHEeY7WwChML6aXZwf2q919NQ=</Signature></License>"

filename_ = os.path.basename(__file__).replace('.py', '')
default_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\knn_model_n_8_classes_2_data_grasping_ungrasping_5_20250418_1636_window_200_step_50_20250418_175135.pkl'
# default_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\knn_model_n_20_classes_3_data_relaxing_grasping_opening_window_200_step_50_20250422_172557.pkl'
# default_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\knn_model_n_20_classes_3_data_relaxing_grasping_opening_all_data_window_200_step_50_20250422_182814.pkl'

# RF:
# default_rf_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_100_classes_2_data_relaxing_grasping_opening_window_100_step_50_20250424_135118.pkl'
# Best:
default_rf_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_100_classes_2_data_relaxing_grasping_opening_window_200_step_50_20250424_131144.pkl'
# default_rf_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_100_classes_2_data_relaxing_grasping_window_200_step_5_20250515_pq171019.pkl'
# default_rf_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_10_classes_2_data_relaxing_grasping_window_200_step_5_20250515_171427.pkl'
# default_rf_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_100_classes_3_data_relaxing_grasping_opening_window_200_step_50_20250515_154319.pkl'

# Relaxing and Grasping trained as one class:
# default_rf_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_100_classes_2_data_relaxing_grasping_window_200_step_50_20250515_164630.pkl'

# Time features only:
# default_rf_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_100_classes_2_data_relaxing_grasping_window_100_step_50_20250515_151112.pkl'
# default_rf_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_100_classes_3_data_relaxing_grasping_opening_window_200_step_50_20250515_152731.pkl'

default_mlp_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\mlp_model_act_relu_classes_2_data_relaxing_grasping_window_200_step_5_20250516_121338.pkl'
default_mlp_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\mlp_model_act_relu_classes_3_data_relaxing_grasping_opening_window_200_step_5_20250516_124235.pkl'
# default_mlp_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\mlp_model_act_relu_classes_2_data_relaxing_grasping_opening_window_200_step_5_20250516_124558.pkl'


class TrignoBase():
    """
    Note: Based on thr TrignoBase provided in the Example-Applications repository.

    AeroPy reference imported above then instantiated in the constructor below
    All references to TrigBase. call an AeroPy method (See AeroPy documentation for details)
    """

    def __init__(self, csv_file_path="data.csv",
                 no_classification=False, debug=False):
        self.TrigBase = AeroPy()
        self.channel_guids = []
        self.channelcount = 0
        self.pairnumber = 0
        self.csv_writer = CsvWriter(filename=csv_file_path)
        
        self.streamYTData = False
        self.pauseFlag = True

        self.no_classification = no_classification
        self.debug = debug
        
    def init_classifier(self):
        ## ----------------------------------------------------------------------
        ## Classifier Initialization
        ## ----------------------------------------------------------------------

        print(f'[INFO] [{filename_}] Initializing classifier...')
        # self.classifier_ = EMGClassifier(model_file_path=default_model_path_,
        #                                                     model_type='knn',
        #                                                     window_size=200,
        #                                                     window_sliding_step=200)
        #                                                     # window_sliding_step=50)
        # self.classifier_ = EMGClassifier(model_file_path=default_rf_model_path_,
        #                                                     model_type='rf',
        #                                                     window_size=200,
        #                                                     window_sliding_step=200)
        
        # self.classifier_ = EMGClassifier(model_spec_dicts=[{'model_type': 'knn', 'model_file_path': default_model_path_}],
        #                                                     window_size=200,
        #                                                     window_sliding_step=200)
        # self.classifier_ = EMGClassifier(model_spec_dicts=[{'model_type': 'rf', 'model_file_path': default_rf_model_path_}],
        #                                                     window_size=200,
        #                                                     window_sliding_step=200)
        # self.classifier_ = EMGClassifier(model_spec_dicts=[{'model_type': 'rf', 'model_file_path': default_rf_model_path_},
        #                                                    {'model_type': 'rf', 'model_file_path': 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_10_classes_2_data_relaxing_grasping_window_200_step_5_20250515_171427.pkl'}],
        #                                                     window_size=200,
        #                                                     window_sliding_step=200)
        # self.classifier_ = EMGClassifier(model_spec_dicts=[{'model_type': 'mlp', 'model_file_path': default_mlp_model_path_}],
        #                                                     window_size=200,
        #                                                     window_sliding_step=200)
        self.classifier_ = EMGClassifier(model_spec_dicts=[{'model_type': 'rf', 'model_file_path': default_rf_model_path_},
                                                           {'model_type': 'rf', 'model_file_path': 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_10_classes_2_data_relaxing_grasping_window_200_step_5_20250515_171427.pkl'},
                                                           {'model_type': 'knn', 'model_file_path': default_model_path_}],
                                                            window_size=200,
                                                            window_sliding_step=200)
        print(f'[INFO] [{filename_}] Loading pretrained KNN model')
        self.classifier_.load_model()
        
        # TODO: Remove hard-coded value:
        self.num_channels = 4
        self.data_point_limit = self.classifier_.window_size
        self.data_deques = [deque(maxlen=self.data_point_limit) for _ in range(self.num_channels)]
        self.current_t, self.iteration = 0.0, 0
        # TODO: get following from sensor mode:
        self.dt = 0.001
        
        # self.class_ids = self.classifier_.model.classes_
        self.class_ids = self.classifier_.classes
        # Note: for now, assuming max. three classes:
        self.class_colors = dict(zip(self.class_ids, ['grey', 'green', 'purple']))

    def init_udp_interface(self):
        ## ----------------------------------------------------------------------
        ## UDP Parameters and Initialization
        ## ----------------------------------------------------------------------

        # frankaNUC:
        self.udp_ip = '10.157.174.66'
        # NeuroNUC:
        # self.udp_ip = '10.157.174.176'
        self.udp_output_port = 7000
        self.udp_output_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # -- AeroPy Methods --
    def PipelineState_Callback(self):
        return self.TrigBase.GetPipelineState()

    def Connect_Callback(self):
        """Callback to connect to the base"""
        self.TrigBase.ValidateBase(key, license)

    def Start_Callback(self):
        """Callback to start the data stream from Sensors"""

        configured = self.ConfigureCollectionOutput()
        if configured:
            #(Optional) To get YT data output pass 'True' to Start method
            self.TrigBase.Start(self.streamYTData)

    def ConfigureCollectionOutput(self):
        self.pauseFlag = False

        self.DataHandler.packetCount = 0
        self.DataHandler.allcollectiondata = []

        # Pipeline Armed when TrigBase.Configure already called.
        # This if block allows for sequential data streams without reconfiguring the pipeline each time.
        # Reset output data structure before starting data stream again
        if self.TrigBase.GetPipelineState() == 'Armed':
            self.csv_writer.cleardata()
            for i in range(len(self.channelobjects)):
                self.DataHandler.allcollectiondata.append([])
            return True

        # Pipeline Connected when sensors have been scanned in sucessfully.
        # Configure output data using TrigBase.Configure and pass args if you are using a start and/or stop trigger
        elif self.TrigBase.GetPipelineState() == 'Connected':
            self.csv_writer.clearall()
            self.channelcount = 0
            self.TrigBase.Configure(False, False)
            configured = self.TrigBase.IsPipelineConfigured()
            if configured:
                self.channelobjects = []
                self.plotCount = 0
                self.emgChannelsIdx = []
                globalChannelIdx = 0
                self.channel_guids = []

                for i in range(self.SensorCount):

                    selectedSensor = self.TrigBase.GetSensorObject(i)
                    print("(" + str(selectedSensor.PairNumber) + ") " + str(selectedSensor.FriendlyName))

                    # CSV Export Config
                    self.csv_writer.appendSensorHeader(selectedSensor)

                    if len(selectedSensor.TrignoChannels) > 0:
                        print("--Channels")

                        for channel in range(len(selectedSensor.TrignoChannels)):
                            ch_object = selectedSensor.TrignoChannels[channel]
                            if str(ch_object.Type) == "SkinCheck":
                                continue

                            ch_guid = ch_object.Id
                            ch_type = str(ch_object.Type)

                            get_all_channels = True
                            if get_all_channels:
                                self.channel_guids.append(ch_guid)
                                globalChannelIdx += 1

                                #CSV Export Config
                                if not self.streamYTData:
                                    self.csv_writer.appendChannelHeader(ch_object)
                                    if channel > 0 & channel != len(selectedSensor.TrignoChannels):
                                        self.csv_writer.appendSensorHeaderSeperator()
                                else:
                                    self.csv_writer.appendYTChannelHeader(ch_object)
                                    if channel == 0:
                                        self.csv_writer.appendSensorHeaderSeperator()
                                    elif channel > 0 & channel != len(selectedSensor.TrignoChannels):
                                        self.csv_writer.appendYTSensorHeaderSeperator()

                            #NOTE: The self.channel_guids list is used to parse select channels during live data streaming in DataManager.py
                            #      this example will add all available channels to this list (above)
                            #      if you want to only parse certain channels then add only those channel guids to this list
                            #      for example: if you only want the EMG channels during live data streaming (flip bool above to false):
                            if not get_all_channels:
                                if ch_type == 'EMG':
                                    self.channel_guids.append(ch_guid)
                                    self.csv_writer.h2_channels.append(
                                        ch_object.Name + " (" + str(ch_object.SampleRate) + ")")
                                    if channel > 0:
                                        self.csv_writer.h1_sensors.append(",")
                                    globalChannelIdx += 1

                            sample_rate = round(selectedSensor.TrignoChannels[channel].SampleRate, 3)
                            print("----" + selectedSensor.TrignoChannels[channel].Name + " (" + str(sample_rate) + " Hz) " + str(selectedSensor.TrignoChannels[channel].Id))
                            self.channelcount += 1
                            self.channelobjects.append(channel)
                            self.DataHandler.allcollectiondata.append([])

                            # NOTE: Plotting/Data Output: This demo does not plot non-EMG channel types such as
                            # accelerometer, gyroscope, magnetometer, and others. However, the data from channels
                            # that are excluded from plots are still available via output from PollData()

                            # ---- Plot EMG Channels
                            if ch_type == 'EMG':
                                self.emgChannelsIdx.append(globalChannelIdx-1)
                                self.plotCount += 1

                            # ---- Exclude non-EMG channels from plots
                            else:
                                pass

                return True
        else:
            return False

    def Scan_Callback(self):
        """Callback to tell the base to scan for any available sensors"""
        try:
            f = self.TrigBase.ScanSensors().Result
        except Exception as e:
            print("Python demo attempt another scan...")
            time.sleep(1)
            self.Scan_Callback()

        self.all_scanned_sensors = self.TrigBase.GetScannedSensorsFound()
        print("Sensors Found:\n")
        for sensor in self.all_scanned_sensors:
            print("(" + str(sensor.PairNumber) + ") " +
                sensor.FriendlyName + "\n" +
                sensor.Configuration.ModeString + "\n")

        self.SensorCount = len(self.all_scanned_sensors)
        for i in range(self.SensorCount):
            self.TrigBase.SelectSensor(i)
           
            # Set to desired mode:
            sample_mode_str = 'EMG raw x4 (1111Hz), +/-5.5mV, 20-450Hz'
            print(f'[INFO] Setting sample mode to: {sample_mode_str}')
            self.setSampleMode(i, sample_mode_str)
            print(f'[INFO] Current sample mode: {self.getCurMode(i)}')

        return self.all_scanned_sensors
    
    def getSampleModes(self, sensorIdx):
        """Gets the list of sample modes available for selected sensor"""
        sampleModes = self.TrigBase.AvailibleSensorModes(sensorIdx)
        return sampleModes

    def getCurMode(self, sensorIdx):
        """Gets the current mode of the sensors"""
        if sensorIdx >= 0 and sensorIdx <= self.SensorCount:
            curModes = self.TrigBase.GetCurrentSensorMode(sensorIdx)
            return curModes
        else:
            return None

    def setSampleMode(self, curSensor, setMode):
        """Sets the sample mode for the selected sensor"""
        self.TrigBase.SetSampleMode(curSensor, setMode)
        mode = self.getCurMode(curSensor)
        sensor = self.TrigBase.GetSensorObject(curSensor)
        if mode == setMode:
            print("(" + str(sensor.PairNumber) + ") " + str(sensor.FriendlyName) +" Mode Change Successful")

            print(f'[DEBUG] Current mode: {mode}')
            
    ## CollectDataController relevant functions:

    def threadManager(self, start_trigger, stop_trigger):
        """Handles the threads for the DataCollector gui"""
        self.emg_queue = deque(maxlen=1000)

        # Start standard data stream (only channel data, no time values)
        if not self.streamYTData:
            self.t1 = threading.Thread(target=self.streaming)
            self.t1.daemon = True
            self.t1.start()
            
            if self.no_classification:
                print(f'[WARN] [{filename_}] Not starting classification thread!')
            else:
                self.tc = threading.Thread(target=self.classification_thread)
                self.tc.daemon = True
                self.tc.start()

    def streaming(self):
        """This is the data processing thread"""
        self.emg_queue = deque(maxlen=1000)
        while self.pauseFlag is True:
            continue
        while self.pauseFlag is False:
            # if self.debug:
            #     new_data_point_start_time = time.time()
            self.DataHandler.processData(self.emg_queue)
            # if self.debug:
            #     print(f'[DEBUG] [{filename_}] DataHandler elapsed time: \n{time.time() - new_data_point_start_time:.10f}')
    
    def classification_thread(self):
            """Classification thread"""
            
            streaming_start_time = time.time()
            new_data_point_start_time = time.time()
            classification_start_time = time.time()
            time_elapsed_since_last_classification = 0.
            
            print(f'[INFO] [{filename_}] Starting classification thread...')
            
            while self.pauseFlag is True:
                streaming_start_time = time.time()
                continue
            while self.pauseFlag is False:
                if self.debug:
                    print(f'[DEBUG] [{filename_}] Current elapsed time: {time.time() - streaming_start_time:.5f}')
                while_loop_iteration_start_time = time.time()
                time_elapsed_since_last_classification = ((time.time() - classification_start_time) * 1000)
                
                # Note: emg_queue is a deque of length num_samples elements, each element is a list of num_channels
                # elements, each of which contain the channel's last 30 readings.
                if len(self.emg_queue) > 0:
                    if self.debug:
                        print(f'[DEBUG] [{filename_}] Class. Thread: Time elapsed since last new data batch: \n{(time.time() - new_data_point_start_time) * 1000:.2f}ms')
                        print(f'[DEBUG] [{filename_}] self.iteration: {self.iteration}')
                        new_data_point_start_time = time.time()

                    ## Grab latest data point from sensor:
                    if self.debug:
                        point_popping_start_time = time.time()
                    incData = self.emg_queue.popleft()
                    ## incData = self.emg_queue[-1]; self.emg_queue.clear()
                    if self.debug:
                        print(f'[DEBUG] [{filename_}] Single data point popping time: \n{time.time() - point_popping_start_time:.10f}')

                    ## Populate data queue with latest point:
                    if self.debug:
                        data_deque_population_start_time = time.time()
                    try:
                        for channel_index in range(self.num_channels):
                            # self.data_deques[channel_index].append(incData[channel_index][-1])
                            self.data_deques[channel_index].extend(incData[channel_index].tolist())
                    except IndexError:
                        print(f'[WARN] [{filename_}] Encountered index -1 error. Skipping processing...')
                        continue
                    if self.debug:
                        print(f'[DEBUG] [{filename_}] Data deque population time: \n{time.time() - data_deque_population_start_time:.10f}')

                    self.iteration += 1
                    
                    predicted_window_class = None
                    
                    if self.debug:
                        print(f'[DEBUG] [{filename_}] Time elapsed since last classification: {round(time_elapsed_since_last_classification)}ms')

                    ## Run classification on latest data window:
                    # if (self.iteration % self.classifier_.window_sliding_step) == 0 and self.iteration != 0:
                    if round(time_elapsed_since_last_classification) > self.classifier_.window_sliding_step:
                        classification_start_time = time.time()
                        time_elapsed_since_last_classification = 0.
                        
                        self.window_values_df = pd.DataFrame(columns=self.emgChannelsIdx)
                        for i in self.emgChannelsIdx:
                            self.window_values_df[i] = self.data_deques[i]

                        if len(self.window_values_df) != self.classifier_.window_size:
                            print(f'[WARN] [{filename_}] Got data segment of size {len(self.window_values_df)} instead of ' + \
                                f'window size {self.classifier_.window_size}! Skipping computation...')
                            predicted_window_class = None
                        else:
                            if self.debug:
                                feature_extraction_start_time = time.time()
                            features_vector = self.classifier_.get_features_vector(self.window_values_df)
                            # features_vector = self.classifier_.get_time_features_vector(self.window_values_df)

                            if self.debug:
                                print(f'[DEBUG] [{filename_}] Feature extraction time: {time.time() - feature_extraction_start_time}')
                                inference_start_time = time.time()

                            # predicted_window_class = self.classifier_.model.predict(features_vector.reshape(-1, 1).T).item()
                            # predicted_window_class = self.classifier_.predict(features_vector.reshape(-1, 1).T).item()
                            
                            try:
                                predicted_window_class = self.classifier_.predict(features_vector.reshape(-1, 1).T)
                            except ValueError:
                                print(f'[WARN] [{filename_}] Possible got NaN values in feature vector. ' + \
                                      f'Skipping computation...')
                                continue
                            if predicted_window_class is None:
                                continue
                            # else:
                            #     predicted_window_class = predicted_window_class.item()
                            if self.debug:
                                print(f'[DEBUG] [{filename_}] Inference time: {time.time() - inference_start_time}')
                                print(f'[DEBUG] [{filename_}] Predicted class: {predicted_window_class}')
                                print()

                            # Send result over UDP message:
                            message = {'class': predicted_window_class, 'timestamp': time.time()}
                            message = json.dumps(message, indent = 4)
                            if self.debug:
                                print(f'[DEBUG] [{filename_}] Sending UDP message: \n{message}')
                            message = message.encode()
                            self.udp_output_socket.sendto(message, (self.udp_ip, self.udp_output_port))

                            if self.debug:
                                print(f'[DEBUG] [{filename_}] Full classification (w/ UDP) time: {time.time() - classification_start_time}')
                    
                    if self.debug:
                        print(f'[DEBUG] [{filename_}] Single data point processing time: \n{time.time() - while_loop_iteration_start_time:.10f}')
                
                if self.debug:
                    print(f'[DEBUG] [{filename_}] Full loop iteration time: \n{time.time() - while_loop_iteration_start_time:.3f}')
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('-f','--csv_file_path', type=str,
                        default="C:\\Users\\go98voq\\automatica_2025_win\\Example-Applications\\trial_data\\data.csv",
                        help='Absolute path to EMG data CSV file to be saved')
    parser.add_argument('--no_classification', dest='no_classification', action='store_true')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(no_classification=False, debug=False)
    args = parser.parse_args()
    
    base = TrignoBase(csv_file_path=args.csv_file_path,
                      no_classification=args.no_classification,
                      debug=args.debug)
    DataHandler = DataKernel(base)
    base.DataHandler = DataHandler
    
    base.Connect_Callback()
    sensorList = base.Scan_Callback()
    
    if not args.no_classification:
        print(f'[INFO] [{filename_}] Excluding classification!')
        base.init_classifier()
    base.init_udp_interface()
    
    pipelinetext = base.PipelineState_Callback()
    print(f'[INFO] [{filename_}] Pipeline state: {pipelinetext}')

    print(f'[INFO] [{filename_}] Press the "b" key to start data collection and classification')
    print(f'[INFO] [{filename_}] Press the "p" key to pause data collection and classification')
    print(f'[INFO] [{filename_}] Press the "q" key to stop the program')
    paused = True
    try:
        while True:
            if paused and keyboard.is_pressed('q'):
                print(f'[INFO] [{filename_}] Stopping...')
                break

            if not paused and keyboard.is_pressed('p'):
                paused = True
                print(f'[INFO] [{filename_}] Pausing data collection')
                base.pauseFlag = True
                base.TrigBase.Stop()
                base.csv_writer.data = base.DataHandler.allcollectiondata
                
                print()
                print(f'[INFO] [{filename_}] Press the "b" key to start data collection and classification')
                print(f'[INFO] [{filename_}] Press the "p" key to pause data collection and classification')
                print(f'[INFO] [{filename_}] Press the "e" key to export collected data')
                print(f'[INFO] [{filename_}] Press the "q" key to stop the program')

            if paused and keyboard.is_pressed('b'):
                paused = False
                configured = base.ConfigureCollectionOutput()
                if configured:
                    base.TrigBase.Start(base.streamYTData)
                    base.threadManager(False, False)
                print(f'[INFO] [{filename_}] Starting Data collection...')
                
            if paused and keyboard.is_pressed('e'):
                export = None
                if base.streamYTData:
                    export = base.csv_writer.exportYTCSV()
                else:
                    export = base.csv_writer.exportCSV()
                if export:
                    print(f'[INFO] [{filename_}] Exporting data to CSV file: {args.csv_file_path}...')
                time.sleep(1)
        
            time.sleep(0.01)
    except (KeyboardInterrupt):
        print(f'[INFO] [{filename_}] Stopping loop')
        
    print(f'[INFO] [{filename_}] Ended')