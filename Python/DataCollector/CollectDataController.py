"""
Controller class for the Data Collector GUI
This is the controller for the GUI that lets you connect to a base, scan via rf for sensors, and stream data from them in real time.
"""

from collections import deque

from Plotter.GenericPlot import *
from AeroPy.TrignoBase import *
from AeroPy.DataManager import *

clr.AddReference("System.Collections")

app.use_app('PySide6')

import os
import json
import socket
import pandas as pd
from franka_emg_grasping import emg_classification
from franka_emg_grasping.emg_classification_refactored import EMGClassifier

filename_ = os.path.basename(__file__).replace('.py', '')

# KNN:
default_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\knn_model_n_8_classes_2_data_grasping_ungrasping_5_20250418_1636_window_200_step_50_20250418_175135.pkl'
# default_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\knn_model_n_20_classes_3_data_relaxing_grasping_opening_window_200_step_50_20250422_172557.pkl'
# default_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\knn_model_n_20_classes_3_data_relaxing_grasping_opening_all_data_window_200_step_50_20250422_182814.pkl'

# RF:
# default_rf_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_100_classes_2_data_grasping_ungrasping_5_20250418_1636_window_200_step_50_20250424_111228.pkl'
# default_rf_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_100_classes_3_data_relaxing_grasping_opening_window_200_step_50_20250424_120054.pkl'
# default_rf_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_1000_classes_3_data_relaxing_grasping_opening_window_200_step_50_20250424_122046.pkl'
# default_rf_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_1000_classes_2_data_relaxing_grasping_opening_window_200_step_50_20250424_123422.pkl'
# default_rf_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_1000_classes_2_data_relaxing_grasping_opening_window_200_step_50_20250424_130525.pkl'
# default_rf_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_100_classes_2_data_relaxing_grasping_opening_window_200_step_50_20250424_131144.pkl'
# default_rf_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_100_classes_2_data_relaxing_grasping_opening_window_50_step_50_20250424_132142.pkl'
default_rf_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_100_classes_2_data_relaxing_grasping_opening_window_100_step_50_20250424_135118.pkl'

# ## Current best (UPDATE: before data point fix):
# self.classifier_ = emg_classification.EMGClassifier(model_file_path='C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_100_classes_2_data_relaxing_grasping_opening_window_200_step_50_20250424_131144.pkl',
#                                                             model_type='rf',
#                                                             window_size=100,
#                                                             window_sliding_step=50)

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

# default_mlp_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\mlp_model_act_relu_classes_2_data_relaxing_grasping_window_200_step_5_20250516_121338.pkl'
# default_mlp_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\mlp_model_act_relu_classes_3_data_relaxing_grasping_opening_window_200_step_5_20250516_124235.pkl'
# default_mlp_model_path_ = 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\mlp_model_act_relu_classes_2_data_relaxing_grasping_opening_window_200_step_5_20250516_124558.pkl'


class PlottingManagement():
    def __init__(self, collect_data_window, metrics, emgplot=None, with_classifications_indicator=False, debug=False):
        self.base = TrignoBase(self)
        self.collect_data_window = collect_data_window
        self.EMGplot = emgplot
        self.metrics = metrics
        self.packetCount = 0  # Number of packets received from base
        self.pauseFlag = True  # Flag to start/stop collection and plotting
        self.DataHandler = DataKernel(self.base)  # Data handler for receiving data from base
        self.base.DataHandler = self.DataHandler
        self.outData = [[0]]
        self.Index = None
        self.newTransform = None

        self.streamYTData = False # set to True to stream data in (T, Y) format (T = time stamp in seconds Y = sample value)

        ## ----------------------------------------------------------------------
        ## Classifier Initialization
        ## ----------------------------------------------------------------------

        print(f'[INFO] [{filename_}] Initializing classifier...')
        # self.classifier_ = emg_classification.EMGClassifier(model_file_path='C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_100_classes_2_data_relaxing_grasping_opening_window_200_step_50_20250424_131144.pkl',
        #                                                             model_type='rf',
        #                                                             window_size=100,
        #                                                             window_sliding_step=50)
        # self.classifier_ = emg_classification.EMGClassifier(model_file_path='C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_100_classes_2_data_relaxing_grasping_opening_window_100_step_50_20250424_135118.pkl',
        #                                                     model_type='rf',
        #                                                     window_size=100,
        #                                                     window_sliding_step=50)
        self.classifier_ = EMGClassifier(model_spec_dicts=[{'model_type': 'rf', 'model_file_path': default_rf_model_path_},
                                                           {'model_type': 'rf', 'model_file_path': 'C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\rf_model_n_10_classes_2_data_relaxing_grasping_window_200_step_5_20250515_171427.pkl'},
                                                           {'model_type': 'knn', 'model_file_path': default_model_path_}],
                                                            window_size=200,
                                                            window_sliding_step=200)
        print(f'[INFO] [{filename_}] Loading pretrained model')
        self.classifier_.load_model()
        
        # self.classifier_2_ = emg_classification.EMGClassifier(model_file_path="C:\\Users\\go98voq\\automatica_2025_win\\franka-emg-grasping\\dev\\output_data\\knn_model_n_8_classes_2_data_grasping_ungrasping_5_20250418_1643_window_200_step_50_20250418_172419.pkl",
        #                                                     model_type='knn',
        #                                                     window_size=200,
        #                                                     window_sliding_step=50)
        # self.classifier_2_.load_model()
        
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
        
        ## ----------------------------------------------------------------------
        ## UDP Parameters and Initialization
        ## ----------------------------------------------------------------------

        # frankaNUC:
        self.udp_ip = '10.157.174.66'
        # NeuroNUC:
        # self.udp_ip = '10.157.174.176'
        self.udp_output_port = 7000
        self.udp_output_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.with_classifications_indicator = with_classifications_indicator
        self.debug = debug

    def streaming(self):
        """This is the data processing thread"""
        self.emg_queue = deque(maxlen=1000)
        new_data_point_start_time = time.time()
        while self.pauseFlag is True:
            continue
        while self.pauseFlag is False:
            while_loop_iteration_start_time = time.time()
            # if self.debug:
            #     new_data_point_start_time = time.time()
            self.DataHandler.processData(self.emg_queue)
            # if self.debug:
            #     print(f'[DEBUG] [{filename_}] DataHandler elapsed time: \n{time.time() - new_data_point_start_time:.10f}')
            
            # Note: emg_queue is a deque of length num_samples elements, each element is a list of num_channels
            # elements, each of which contain the channel's last 30 readings.
            # print(f'[DEBUG] len(self.emg_queue): {len(self.emg_queue)}')

            # if len(self.emg_queue) > 0:
            #     incData = self.emg_queue.popleft()  # Data at time T-1
            #     print(f'[DEBUG] New data point elapsed time: \n{time.time() - new_data_point_start_time:.10f}')
            #     new_data_point_start_time = time.time()

            #     print(f'[DEBUG] len(incData): {len(incData)}')
            #     print(f'[DEBUG] incData[0].shape: {incData[0].shape}')
            #     print(f'[DEBUG] incData[all_channels][-1]: {[incData[channel_index][-1] for channel_index in self.base.emgChannelsIdx]}')
            #     # print(f'[DEBUG] incData[0][-1]: {incData[0][-1]}')
            #     print(f'[DEBUG] incData: {incData}')

            self.updatemetrics()

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
                # if self.with_classifications_indicator and classification_start_time is not None and (time.time() - classification_start_time > 1.0):
                #     self.collect_data_window.classifications_window.set_color()
                
                if self.debug:
                    print(f'[DEBUG] [{filename_}] Time elapsed since last classification: {round(time_elapsed_since_last_classification)}ms')

                ## Run classification on latest data window:
                # if (self.iteration % self.classifier_.window_sliding_step) == 0 and self.iteration != 0:
                if round(time_elapsed_since_last_classification) > self.classifier_.window_sliding_step:
                    classification_start_time = time.time()
                    time_elapsed_since_last_classification = 0.
                    
                    self.window_values_df = pd.DataFrame(columns=self.base.emgChannelsIdx)
                    for i in self.base.emgChannelsIdx:
                        self.window_values_df[i] = self.data_deques[i]

                    if len(self.window_values_df) != self.classifier_.window_size:
                        print(f'[WARN] [{filename_}] Got data segment of size {len(self.window_values_df)} instead of ' + \
                              f'window size {self.classifier_.window_size}! Skipping computation...')
                        predicted_window_class = None
                    else:
                        if self.debug:
                            feature_extraction_start_time = time.time()
                        features_vector = self.classifier_.get_features_vector(self.window_values_df)

                        if self.debug:
                            print(f'[DEBUG] [{filename_}] Feature extraction time: {time.time() - feature_extraction_start_time}')
                            inference_start_time = time.time()

                        # predicted_window_class = self.classifier_.model.predict(features_vector.reshape(-1, 1).T).item()
                        # predicted_window_class_2 = self.classifier_2_.model.predict(features_vector.reshape(-1, 1).T).item()
                        
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
                            
                        # Set classification window to predicted class color
                        if self.with_classifications_indicator:
                            self.collect_data_window.classifications_window.set_color(color=self.class_colors[predicted_window_class])
                        
                        if self.debug:
                            print(f'[DEBUG] [{filename_}] Full classification (w/ UDP) time: {time.time() - classification_start_time}')
                
                if self.debug:
                    print(f'[DEBUG] [{filename_}] Single data point processing time: \n{time.time() - while_loop_iteration_start_time:.10f}')
            
            if self.debug:
                print(f'[DEBUG] [{filename_}] Full loop iteration time: \n{time.time() - while_loop_iteration_start_time:.3f}')

    def streamingYT(self):
        """This is the data processing thread"""
        self.emg_queue = deque()
        while self.pauseFlag is True:
            continue
        while self.pauseFlag is False:
            self.DataHandler.processYTData(self.emg_plot)
            self.updatemetrics()

    def vispyPlot(self):
        """Plot Thread - Only Plotting EMG Channels"""
        while self.pauseFlag is False:
            if len(self.emg_plot) >= 2:
                incData = self.emg_plot.popleft()  # Data at time T-1
                try:
                    self.outData = list(np.asarray(incData, dtype='object')[tuple([self.base.emgChannelsIdx])])
                except IndexError:
                    print("Index Error Occurred: vispyPlot()")
                if self.base.emgChannelsIdx and len(self.outData[0]) > 0:
                    try:
                        self.EMGplot.plot_new_data(self.outData,
                                                   [self.emg_plot[0][i][0] for i in self.base.emgChannelsIdx])
                    except IndexError:
                        print("Index Error Occurred: vispyPlot()")

    def updatemetrics(self):
        self.metrics.framescollected.setText(str(self.DataHandler.packetCount))

    def resetmetrics(self):
        self.metrics.framescollected.setText("0")
        self.metrics.totalchannels.setText(str(self.base.channelcount))

    def threadManager(self, start_trigger, stop_trigger):
        """Handles the threads for the DataCollector gui"""
        self.emg_plot = deque()

        # Start standard data stream (only channel data, no time values)
        if not self.streamYTData:
            self.t1 = threading.Thread(target=self.streaming)
            self.t1.start()
            
            self.tc = threading.Thread(target=self.classification_thread)
            self.tc.start()

        # Start YT data stream (with time values)
        else:
            self.t1 = threading.Thread(target=self.streamingYT)
            self.t1.start()

        if self.EMGplot:
            self.t2 = threading.Thread(target=self.vispyPlot)
            if not start_trigger:
                self.t2.start()

        if start_trigger:
            self.t3 = threading.Thread(target=self.waiting_for_start_trigger)
            self.t3.start()

        if stop_trigger:
            self.t4 = threading.Thread(target=self.waiting_for_stop_trigger)
            self.t4.start()

    def waiting_for_start_trigger(self):
        while self.base.TrigBase.IsWaitingForStartTrigger():
            continue
        self.pauseFlag = False
        if self.EMGplot:
            self.t2.start()
        print("Trigger Start - Collection Started")

    def waiting_for_stop_trigger(self):
        while self.base.TrigBase.IsWaitingForStartTrigger():
            continue
        while self.base.TrigBase.IsWaitingForStopTrigger():
            continue
        self.pauseFlag = True
        self.metrics.pipelinestatelabel.setText(self.base.PipelineState_Callback())
        self.base.csv_writer.data = self.DataHandler.allcollectiondata
        self.collect_data_window.exportcsv_button.setEnabled(True)
        self.collect_data_window.exportcsv_button.setStyleSheet("color : white")
        print("Trigger Stop - Data Collection Complete")
        self.DataHandler.processData(self.emg_plot)
