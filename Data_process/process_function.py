from logging import root
import os
import sys

import numpy as np
import scipy.linalg
import scipy.io
import scipy.sparse


current_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(current_path)[0]

sys.path.append(current_path)
import LoadData

def Load_BCIC_2a_raw_data(data_path = None,tmin=0, tmax=4,bandpass = [0,38],resample = None):
    '''
    Load the BCIC 2a data.
    Arg:
        sub:The subject whose data need to be load.
    '''
    if data_path is None:
        # Use the default path.
        data_path = os.path.join(root_path,'Raw_data','BCICIV_2a_gdf') 

    if not os.path.exists(data_path):
        print('Please download the BCICIV_2a data first! and unzip it to {}'.format(data_path))
        exit(0)

    for sub in range(1,10):
        data_name = r'A0{}T.gdf'.format(sub)
        data_loader = LoadData.LoadBCIC(data_name, data_path)
        data = data_loader.get_epochs(tmin=tmin, tmax=tmax,bandpass = bandpass,resample = resample)
        print('orgin size is:',data['x_data'].shape)
        train_x = np.array(data['x_data'])[:, :, :]
        train_y = np.array(data['y_labels'])
        
        data_name = r'A0{}E.gdf'.format(sub)
        label_name = r'A0{}E.mat'.format(sub)
        data_loader = LoadData.LoadBCIC_E(data_name, label_name, data_path)
        data = data_loader.get_epochs(tmin=tmin, tmax=tmax,bandpass = bandpass,resample = resample)
        test_x = np.array(data['x_data'])[:, :, :]
        test_y = data['y_labels']
        
        train_x = np.array(train_x)
        train_y = np.array(train_y).reshape(-1)

        test_x = np.array(test_x)
        test_y = np.array(test_y).reshape(-1)
 
        print('trian_x:',train_x.shape)
        print('train_y:',train_y.shape)
        
        print('test_x:',test_x.shape)
        print('test_y:',test_y.shape)
        
        if bandpass is None:
            SAVE_path = os.path.join(root_path,'Data','BCIC_2a_0_4')
        else:
            SAVE_path = os.path.join(root_path,'Data','BCIC_2a_0_4_{}_{}HZ'.format(bandpass[0],bandpass[1]))

        if not os.path.exists(SAVE_path):
            os.makedirs(SAVE_path)
            
        SAVE_test = os.path.join(SAVE_path,r'sub{}_test'.format(sub))
        SAVE_train = os.path.join(SAVE_path,'sub{}_train'.format(sub))
        
        if not os.path.exists(SAVE_test):
            os.makedirs(SAVE_test)
        if not os.path.exists(SAVE_train):
            os.makedirs(SAVE_train)
            
        scipy.io.savemat(os.path.join(SAVE_train, "Data.mat"), {'x_data': train_x,'y_data': train_y})
        scipy.io.savemat(os.path.join(SAVE_test, "Data.mat"), {'x_data': test_x, 'y_data': test_y})

if __name__ == '__main__':

    data_path = os.path.join(root_path,'Raw_data','BCICIV_2a_gdf') 
    
    Load_BCIC_2a_raw_data(data_path)

  