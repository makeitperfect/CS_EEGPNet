import os
import glob
import numpy as np
import scipy.io as sio
import sys
import mne

'''
    This code is part of the fbcsptoolbox,which is a codebase in the github.
    Thanks for the https://fbcsptoolbox.github.io/ to provie this code.
'''

class LoadData:
    def __init__(self, eeg_file_path: str):
        self.eeg_file_path = eeg_file_path
        self.raw_eeg_subject = None

    def load_raw_data_gdf(self, file_to_load):
        self.raw_eeg_subject = mne.io.read_raw_gdf(os.path.join(self.eeg_file_path,file_to_load))
        return self

    def load_raw_data_mat(self,file_to_load):
        self.raw_eeg_subject = sio.loadmat(self.eeg_file_path + '/' + file_to_load)

    def get_all_files(self, file_path_extension: str = None):
        if file_path_extension:
            return glob.glob(self.eeg_file_path+'/'+file_path_extension)
        return os.listdir(self.eeg_file_path)
    def filter(self,low,high):
        self.raw_eeg_subject.filter(low,high)

class LoadBCIC(LoadData):
    """Subclass of LoadData for loading BCI Competition IV Dataset 2a"""
    def __init__(self, file_to_load, *args):
        self.stimcodes = ['769', '770', '771', '772']
        # self.epoched_data={}
        self.file_to_load = file_to_load
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        self.fs = None
        super(LoadBCIC, self).__init__(*args)

    def get_epochs(self, tmin=-4.5, tmax=5.0, bandpass = None,resample = None,baseline=None,reject = False):
        
        self.load_raw_data_gdf(self.file_to_load)
        raw_data = self.raw_eeg_subject
        events, event_ids = mne.events_from_annotations(raw_data)
        self.fs = raw_data.info.get('sfreq')
        if reject == True:
            reject_events = mne.pick_events(events,[1])
            reject_oneset = reject_events[:,0]/self.fs
            duration = [4]*len(reject_events)
            descriptions = ['bad trial']*len(reject_events)
            blink_annot = mne.Annotations(reject_oneset,duration,descriptions)
            raw_data.set_annotations(blink_annot)
        
        stims =[value for key, value in event_ids.items() if key in self.stimcodes]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=True)
        if bandpass is not None:
            epochs.filter(bandpass[0],bandpass[1],method = 'iir')
            # epochs.resample(128)
        if resample is not None:
            epochs.resample(resample)

        epochs = epochs.drop_channels(self.channels_to_remove)
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        self.x_data = epochs.get_data()*1e6
        # length = len(self.x_data)
        eeg_data={'x_data': self.x_data,
                  'y_labels': self.y_labels,
                  'fs': self.fs
                  }
        return eeg_data
    
class LoadBCIC_E(LoadData):
    """A class to lode the test data of the BICI IV 2a dataset"""
    def __init__(self, file_to_load, lable_name, *args):
        self.stimcodes = ('783')
        # self.epoched_data={}
        self.label_name = lable_name # the path of the test label
        self.file_to_load = file_to_load
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        super(LoadBCIC_E, self).__init__(*args)

    def get_epochs(self, tmin=-4.5, tmax=5.0, bandpass = False,resample = None,baseline=None):
        self.load_raw_data_gdf(self.file_to_load)
        raw_data = self.raw_eeg_subject
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in self.stimcodes]
        
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
        if bandpass is not None:
            epochs.filter(bandpass[0],bandpass[1],method = 'iir')
            # epochs.resample(128)
        if resample is not None:
            epochs.resample(resample)
        epochs = epochs.drop_channels(self.channels_to_remove)
        label_info  = sio.loadmat(os.path.join(self.eeg_file_path,self.label_name))
        #label_info shape:(288, 1)
        self.y_labels = label_info['classlabel'].reshape(-1) -1
        # print(self.y_labels)
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data': self.x_data,
                  'y_labels': self.y_labels,
                  'fs': self.fs}
        return eeg_data
