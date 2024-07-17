import os
import scipy
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import random

root_path = os.path.abspath(os.path.dirname(__file__))

sys.path.append(root_path)
    

class TaskBatchSampler(torch.utils.data.Sampler):
    def __init__(self,dataset,k):
        self.y,self.d = dataset.y,dataset.domain
        self.k = k
        dataset_index = np.array([i for i in range(len(dataset))])

        self.label_index = []
        for i in range(len(self.d.unique())):
        
            domin_index = dataset_index[(self.d==i)]
            domain_y = self.y[domin_index]
            for l in range(len(self.y.unique())):
                # print(len(domin_index[domain_y==l]))
                self.label_index.append(domin_index[domain_y==l])
        # print(len(self.label_index[0]))

    def __iter__(self):
        
        indices_list = [[np.random.choice(i_index,self.k,replace = False).tolist() for i_index in self.label_index] for i in range((len(self.label_index[0])//self.k))]

        for index in indices_list:
            # print(sum(index,[]))
            yield sum(index,[])
    
    def __len__(self):
        return len(self.label_index[0])//self.k

class CustomRandomGroupBatchSampler(torch.utils.data.Sampler):
    def __init__(self, num_groups,group_size,batch_size,batch_shuffle = False):
        self.batch_shuffle = batch_shuffle
        self.batch_size = batch_size
        self.num_groups = num_groups  # 假设有16组数据
        self.group_size = group_size # 假设每组259个数据
        self.num_samples = self.num_groups * self.group_size
        self.groups = list(range(self.num_groups))
    
    def __iter__(self):
        random.shuffle(self.groups)  # 随机化组的顺序
        indices = []
        
        for group in self.groups:
            group_indices = list(range(group * self.group_size, (group + 1) * self.group_size))
            random.shuffle(group_indices)  # 随机化组内数据的顺序
            indices.extend(group_indices)
        
        
        indices_list = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.batch_shuffle is True:
            random.shuffle(indices_list)  # 随机化不同组之间的 batch 顺序

        for index in indices_list:
            yield index
    
    def __len__(self):
        return self.num_samples // self.batch_size

class eeg_dataset(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self,feature,label,domain = None):
        super(eeg_dataset,self).__init__()

        self.x = feature
        self.y = label
        self.domain = domain

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if self.domain is None:
            return self.x[index], self.y[index]
        return self.x[index], self.y[index],self.domain[index]

def get_few_shot_test(sub,data_path,few_shot_number):

    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
    target_session_2_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    session_2_data = sio.loadmat(target_session_2_path)

    x_1 = session_1_data['x_data']
    y_1 = session_1_data['y_data'].reshape(-1)

    x_2 = session_2_data['x_data']
    y_2 = session_2_data['y_data'].reshape(-1)


    x = np.append(x_1,x_2,0)
    y = np.append(y_1,y_2,0)

    # x = x_2
    # y = y_2

    rand_factor = np.random.permutation(len(y))
    x = x[rand_factor]
    y = y[rand_factor]

    few_x = []
    few_y = []

    test_x = []
    test_y = []
    
    for l in np.unique(y):
        few_x.extend(x[y==l][:few_shot_number])
        few_y.extend([l]*few_shot_number)

        test_x.extend(x[y==l][few_shot_number:])
        test_y.extend(y[y==l][few_shot_number:])



    few_x,few_y = torch.FloatTensor(np.array(few_x)),torch.LongTensor(np.array(few_y).reshape(-1))

    test_x,test_y = torch.FloatTensor(np.array(test_x)),torch.LongTensor(np.array(test_y).reshape(-1))


    # print(few_x.shape,test_x.shape)
    # print(few_y.shape,test_y.shape)

    test_dataset = eeg_dataset(test_x,test_y)
    

    return few_x,few_y,test_dataset

def get_data(sub,data_path,sub_num = 9):
    
    # target_y_data = []
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
    target_session_2_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    session_2_data = sio.loadmat(target_session_2_path)
    R = None

    session_1_x = session_1_data['x_data']

    session_2_x = session_2_data['x_data']
    
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)

    test_x_2 = torch.FloatTensor(session_2_x)      
    test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)


    test_x = torch.cat([test_x_1,test_x_2],dim = 0)
    test_y = torch.cat([test_y_1,test_y_2],dim = 0)

    test_dataset = eeg_dataset(test_x,test_y)
  
    
    source_train_x = []
    source_train_y = []
    
    source_valid_x = []
    source_valid_y = []

    source_train_domain = []
    source_valid_domain = []
    
    d = 0
    for i in range(1,sub_num+1):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
    
        test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
        test_data = sio.loadmat(test_path)
 
        session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 0.2,stratify = session_1_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_domain.extend( np.ones_like(train_y)*d)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_domain.extend( np.ones_like(valid_y)*d)

        # d += 1


        session_2_x = test_data['x_data']

        session_2_y = test_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_2_x,session_2_y,test_size = 0.2,stratify = session_2_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_domain.extend(np.ones_like(train_y)*d)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_domain.extend( np.ones_like(valid_y)*d)
        

        d+=1
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_domain = torch.LongTensor(np.array(source_train_domain))

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_domain = torch.LongTensor(np.array(source_valid_domain))
    
    train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_domain)
  
    valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_domain)
    
    return train_dataset,valid_datset,test_dataset

def get_data_woso(sub,data_path):
    # target_y_data = []
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
    target_session_2_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    session_2_data = sio.loadmat(target_session_2_path)
    R = None

    session_1_x = session_1_data['x_data']

    # few_x = torch.FloatTensor(few_x)
    # few_y = torch.LongTensor(few_y)

    session_2_x = session_2_data['x_data']
    
    test_x_1 = torch.FloatTensor(session_2_x)      
    test_y_1 = torch.LongTensor(session_2_data['y_data']).reshape(-1)

    test_x_2 = torch.FloatTensor(session_2_x)      
    test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)

    test_dataset = eeg_dataset(torch.cat([test_x_1,test_x_2],dim=0),torch.cat([test_y_1,test_y_2],dim=0))

    # source_train_x = []
    # source_train_y = []
    
    # source_valid_x = []
    # source_valid_y = []

    source_train_domain = []
    s = [i for i in range(1,10) if i!= sub]
    for j in s:
        source_train_x = []
        source_train_y = []
        
        source_valid_x = []
        source_valid_y = []
        for i in range(1,10):
            if i == sub:
                continue
            train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
            train_data = sio.loadmat(train_path)
        
            test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
            test_data = sio.loadmat(test_path)

            session_1_x = train_data['x_data']

            session_1_y = train_data['y_data'].reshape(-1)

            if i != j:
                source_train_x.extend(session_1_x)
                source_train_y.extend(session_1_y)
                # source_train_domain.extend( np.ones_like(train_y)*d)
            else:
                source_valid_x.extend(session_1_x)
                source_valid_y.extend(session_1_y)


                session_2_x = test_data['x_data']

            session_2_y = test_data['y_data'].reshape(-1)

            if i != j:
                source_train_x.extend(session_2_x)
                source_train_y.extend(session_2_y)
            # source_train_domain.extend( np.ones_like(train_y)*d)
            else:
                source_valid_x.extend(session_2_x)
                source_valid_y.extend(session_2_y)

            
        
        source_train_x = torch.FloatTensor(np.array(source_train_x))
        source_train_y = torch.LongTensor(np.array(source_train_y))
        # source_train_domain = torch.LongTensor(np.array(source_train_domain))

        source_valid_x = torch.FloatTensor(np.array(source_valid_x))
        source_valid_y = torch.LongTensor(np.array(source_valid_y))
        
        train_dataset = eeg_dataset(source_train_x,source_train_y)
    
        valid_datset = eeg_dataset(source_valid_x,source_valid_y)
        

        yield train_dataset,valid_datset,test_dataset

def get_cross_session_data(sub,data_path):

    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
    target_session_2_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    session_2_data = sio.loadmat(target_session_2_path)

    x_1 = session_1_data['x_data']
    y_1 = session_1_data['y_data'].reshape(-1)

    x_2 = session_2_data['x_data']
    y_2 = session_2_data['y_data'].reshape(-1)

    few_x = []
    few_y = []

    class_len = min([sum(y_1 == i) for i in np.unique(y_1)])
    print(class_len)

    for i in np.unique(y_1):
        few_x.extend(x_1[y_1==i][:class_len])
        few_y.extend(y_1[y_1==i][:class_len])
        # few_x.extend(x_1[y_1==i])
        # few_y.extend(y_1[y_1==i])

    few_x,few_y = torch.FloatTensor(np.array(few_x)),torch.LongTensor(np.array(few_y).reshape(-1))

    test_x,test_y = torch.FloatTensor(np.array(x_2)),torch.LongTensor(np.array(y_2).reshape(-1))


    # print(few_x.shape,test_x.shape)
    # print(few_y.shape,test_y.shape)

    test_dataset = eeg_dataset(test_x,test_y)
    

    return few_x,few_y,test_dataset




    
