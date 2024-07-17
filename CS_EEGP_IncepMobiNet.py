import torch
import os
import sys
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
import copy
root_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_path)

from utils import set_seed
import network
from network.utils import FSFN2d

from torch.utils.data import DataLoader
from get_dastaset import *

def get_test_info(model,dataset,batch = None,weight = False,device = 'cuda'):

    if batch is None:
        batch = 72

    data_loader  = DataLoader(dataset,batch_size= batch, shuffle= True,drop_last=False)
    model.eval()

    cor_num = 0.0
    all_loss = 0.0
    with torch.no_grad():
        mean_weight = []
        for data in data_loader:
            x = data[0].to(device)
            y = data[1].to(device)
            predict = model(x)
            y_pred = (-predict).softmax(dim=1)

            # print(predict.shap)
            # loss = F.cross_entropy(predict,y)
            log_p_y = (-y_pred).log_softmax(dim=1)
            loss = F.cross_entropy(log_p_y,y)

            cor_num += torch.sum(torch.argmax(y_pred,dim=1)==y,dtype = torch.float32)
            all_loss += loss

        mean_loss = all_loss/len(data_loader)
        mean_accu = cor_num/len(dataset)
    # print(mean_accu.detach().cpu())
        # print(np.array(mean_weight).var())
    return mean_loss.detach().cpu(),mean_accu.detach().cpu()

def get_online_test_info(few_shot_x,few_shot_y,test_model,dataset,batch = None,device = 'cuda'):

    few_shot_x,few_shot_y = few_shot_x.to(device),few_shot_y.to(device)

    test_model =  copy.deepcopy(test_model)
    # optim = torch.optim.AdamW(test_model.parameters(),lr = 0.0001)
    # optim = torch.optim.AdamW([{'params':test_model.adap_layer.parameters()},{'params':test_model.first_order_feature.parameters()}],lr = 0.0001)
    
    optim = torch.optim.AdamW([{'params':test_model.adap_layer.parameters()}],lr = 0.0001)

    few_shot_loader = DataLoader(eeg_dataset(few_shot_x,few_shot_y),batch_size= 4,shuffle= True)

    loss_fn = torch.nn.NLLLoss().to(device)
    test_model.trans_mode()
    test_model.train()
    # print('开始微调！')
    for i in range(5):
        for x,y in few_shot_loader:
            distances  = test_model(few_shot_x,few_shot_y,x)
            log_p_y = (-distances).log_softmax(dim=1)
            loss = loss_fn(log_p_y,y) 

            optim.zero_grad()
            loss.backward()
            optim.step()


    
    test_model.trans_mode()
    with torch.no_grad():
   
        test_model.eval()
        batch = 32 if batch is None else batch
        data_loader  = DataLoader(dataset,batch_size= batch, shuffle= True,drop_last=False)
        
        cor_num = 0.0
        all_loss = 0.0
        
        mean_weight = []
        for x,y in data_loader:
            x = x.to(device)
            y = y.to(device)
            predict = test_model(few_shot_x,few_shot_y,x)
            # predict = predict[:len(y)]

            y_pred = (-predict).softmax(dim=1)
            
            # print(predict.shap)
            # loss = F.cross_entropy(predict,y)
            log_p_y = (-y_pred).log_softmax(dim=1)

            # print(log_p_y.shape,y.shape)
            loss = F.cross_entropy(log_p_y, y)

            cor_num += torch.sum(torch.argmax(y_pred,dim=1)==y,dtype = torch.float32)
            all_loss += loss

        mean_loss = all_loss/len(data_loader)
        mean_accu = cor_num/len(dataset)
    # test_model.offline()

    return mean_loss.detach().cpu(),mean_accu.detach().cpu()

def get_online_valid_info(test_model,valid_dataset,k_shot = 4,device = 'cuda'):


    valid_loader = DataLoader(valid_dataset,sampler=TaskBatchSampler(valid_dataset,2*k_shot))

    test_model.eval()

    cor_num = 0.0
    all_loss = 0.0

    timer = 0
    with torch.no_grad():
        for x,y,d in valid_loader:
            x,y,d = x[0,:].to(device),y[0,:].to(device),d[0,:].to(device)

            # x = x.reshape(data_info['sub_num']-1,-1,2*k_shot,x.shape[-2],x.shape[-1]).to(device)
            # y = y.reshape(data_info['sub_num']-1,-1,2*k_shot,).to(device)
            # d = d.reshape(data_info['sub_num']-1,-1,2*k_shot,).to(device)

            x = x.reshape(data_info['domain_num'],-1,2*k_shot,x.shape[-2],x.shape[-1]).to(device)
            y = y.reshape(data_info['domain_num'],-1,2*k_shot,).to(device)
            d = d.reshape(data_info['domain_num'],-1,2*k_shot,).to(device)

            for i_x,i_y,i_d in zip(x,y,d):

                s_x,s_y,s_d = i_x[:,:k_shot].flatten(0,1),i_y[:,:k_shot].flatten(0,1),i_d[:,:k_shot].flatten(0,1)
                q_x,q_y,q_d = i_x[:,k_shot:].flatten(0,1),i_y[:,k_shot:].flatten(0,1),i_d[:,k_shot:].flatten(0,1)

                distances = test_model(s_x,s_y,q_x)
                
                y_pred = (-distances).softmax(dim=1)

                log_p_y = (-y_pred).log_softmax(dim=1)
                all_loss += F.cross_entropy(log_p_y, q_y)

                cor_num += torch.sum(torch.argmax(y_pred,dim=1)==q_y,dtype = torch.float32)
                timer += len(q_y)

    mean_loss = all_loss/timer
    mean_accu = cor_num/timer

    return mean_loss,mean_accu

def pre_train(sub,params):
    device = params['device']
    train_dataset,valid_dataset,test_dataset = get_data(sub,data_info['data_path'],data_info['sub_num'])
    print(len(train_dataset),len(valid_dataset),len(test_dataset))
    net = network.__dict__[params['net_name']](**params).to(device)

    ##########################################################
    net.base_mode()

    print('Pretrainning!')

    optim = torch.optim.AdamW(net.parameters(),lr= params['lr'])
    # optim = torch.optim.AdamW(net.parameters(),lr= 0.0006)

    max_valid_accu = .0
    selected_test_accu = .0
    selected_model = None
    dead_epoch = dead_line = params['dead_line']
    loss_fn = torch.nn.NLLLoss().to(device)
    for e in range(params['epoch']):
  

        train_loader = DataLoader(train_dataset,sampler=TaskBatchSampler(train_dataset,params['k_shot']))

        for x,y,d in tqdm(train_loader):
            net.train()
    
            x = x.reshape(data_info['domain_num'],-1,x.shape[-2],x.shape[-1]).to(device)
            y = y.reshape(data_info['domain_num'],-1).to(device)
            d = d.reshape(data_info['domain_num'],-1).to(device)
            
            for data in zip(x,y,d):
                a_x,a_y = data[0],data[1]

                distances = net(a_x)
                log_p_y = (-distances).log_softmax(dim=1)
                loss = loss_fn(log_p_y,a_y) 

                optim.zero_grad()
                loss.backward()
                optim.step()

        valid_loss,valid_accu = get_test_info(net,valid_dataset,64,device = device)

        test_accu = get_test_info(net,test_dataset,16,False,device = device)[1]

        if dead_epoch <= 0:
            break 
        dead_epoch -= 1

        if valid_accu > max_valid_accu:
            dead_epoch = dead_line
            max_valid_accu = valid_accu
            selected_test_accu = test_accu
            selected_model = copy.deepcopy(net.state_dict())

        print('Sub:{}--Epoch:{}--Shot:{}'.format(sub,e,params['k_shot']))
        print('valid_loss:{:.3}--valid_accu:{:.3}--max_valid_accu:{:.3}--dead_epoch:{:}'.format(valid_loss,valid_accu,max_valid_accu,dead_epoch))

        print('test_accu:{:.3}--select_test_accu:{:.3}'.format(test_accu,selected_test_accu))

    save_path = os.path.join(root_path,'Saved_files','trained_model',os.path.basename(__file__)[:-3],'pre_train_({})'.format(params['net_name']),data_info['dataset'],'Seed_{}'.format(params["RANDOM_SEED"]),'sub_{}'.format(sub))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(selected_model,os.path.join(save_path,'model.pth'))
    
    with open(os.path.join(save_path,'info.txt'),'w') as f:
        f.write('The final result is: valid_accu:{}--test_accu:{}'.format(max_valid_accu,selected_test_accu))

    print('The result on test set is:',selected_test_accu)
    return selected_test_accu.detach().cpu()

def train_online(sub,params):
    device = params['device']

    train_dataset,valid_dataset,test_dataset = get_data(sub,data_info['data_path'],data_info['sub_num'])
    print(len(train_dataset),len(valid_dataset),len(test_dataset))
  
    net = network.__dict__[params['net_name']](**params).to(device)

    if params['pretrain_name'] is not None:
        load_path =  os.path.join(root_path,'Saved_files','trained_model',os.path.basename(__file__)[:-3],'pre_train_({})'.format(params['pretrain_name']),data_info['dataset'],'Seed_{}'.format(params["RANDOM_SEED"]),'sub_{}'.format(sub))
    else:
        load_path =  os.path.join(root_path,'Saved_files','trained_model',os.path.basename(__file__)[:-3],'pre_train_({})'.format(params['net_name']),data_info['dataset'],'Seed_{}'.format(params["RANDOM_SEED"]),'sub_{}'.format(sub))

    if not os.path.exists(load_path):
        print('Error,model does not exist！')

    state = torch.load(os.path.join(load_path,'model.pth'),map_location= device)

    net.load_state_dict(state,strict= False)
    print('Model loaded successfully!')

    net.trans_mode()
    max_valid_accu = .0
    selected_test_accu = .0
    selected_model = None
    dead_epoch = dead_line = params['dead_line']

    optim = torch.optim.AdamW(net.parameters(),lr= params['lr'])

    scheduler = torch.optim.lr_scheduler.StepLR(optim,step_size= 4,gamma = 0.9)

    k_shot = params['k_shot']
    # k_shot = 1
    
    loss_fn = torch.nn.NLLLoss().to(device)

    for e in range(params['epoch']):

        train_loader = DataLoader(train_dataset,sampler=TaskBatchSampler(train_dataset,2*k_shot))
        
        for x,y,d in tqdm(train_loader):
            net.train()

            x = x.reshape(data_info['domain_num'],-1,2*k_shot,x.shape[-2],x.shape[-1]).to(device)
            y = y.reshape(data_info['domain_num'],-1,2*k_shot,).to(device)
            d = d.reshape(data_info['domain_num'],-1,2*k_shot,).to(device)

            domain_rand = np.random.permutation(data_info['domain_num'])
            x,y,d = x[domain_rand],y[domain_rand],d[domain_rand]

            for i_x,i_y,i_d in zip(x,y,d):

                s_x,s_y,s_d = i_x[:,:k_shot].flatten(0,1),i_y[:,:k_shot].flatten(0,1),i_d[:,:k_shot].flatten(0,1)
                q_x,q_y,q_d = i_x[:,k_shot:].flatten(0,1),i_y[:,k_shot:].flatten(0,1),i_d[:,k_shot:].flatten(0,1)

                distances = net(s_x,s_y,q_x)
                
                log_p_y = (-distances).log_softmax(dim=1)


                loss = loss_fn(log_p_y, q_y) 
      
                optim.zero_grad()
                loss.backward()
                optim.step()

        scheduler.step()

        valid_losses = []
        valid_accus = []

        for _ in range(10):
            v_l,v_a = get_online_valid_info(net,valid_dataset,4,device = device)
            valid_losses.append(v_l)
            valid_accus.append(v_a)
        valid_loss  = sum(valid_losses)/len(valid_losses)
        valid_accu = sum(valid_accus)/len(valid_accus)


        test_accus = []
        for _ in range(20):
            few_shot_x,few_shot_y, test_dataset = get_few_shot_test(sub,data_info['data_path'],params['k_shot'])
            test_accus.append(get_online_test_info(few_shot_x,few_shot_y,net,test_dataset,device = device)[1])
        test_accu = sum(test_accus)/len(test_accus)


        if dead_epoch <= 0:
            break 
        dead_epoch -= 1

        if valid_accu > max_valid_accu:
            dead_epoch = dead_line
            max_valid_accu = valid_accu
            selected_test_accu = test_accu
            selected_model = copy.deepcopy(net.state_dict())

        print('Sub:{}--Epoch:{}--Shot:{}'.format(sub,e,params['k_shot']))
        print('valid_loss:{:.3}--valid_accu:{:.3}--max_valid_accu:{:.3}--dead_epoch:{:}'.format(valid_loss,valid_accu,max_valid_accu,dead_epoch))

        # print('test_accu:{:.3}--select_test_accu:{:.3}'.format(test_accu,selected_test_accu))

    save_path = os.path.join(root_path,'Saved_files','trained_model',os.path.basename(__file__)[:-3],'train_oneline_({})'.format(params['net_name']),data_info['dataset'],'Seed_{}'.format(params["RANDOM_SEED"]),'sub_{}'.format(sub))
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(selected_model,os.path.join(save_path,'model.pth'))
    
    with open(os.path.join(save_path,'info.txt'),'w') as f:
        f.write('The final result is: valid_accu:{}--test_accu:{}'.format(max_valid_accu,selected_test_accu))

    print('The result on test set is:',selected_test_accu)
    return selected_test_accu.detach().cpu()

def test(sub,params):
    device = params['device']

    net = network.__dict__[params['net_name']](**params).to(device)
 
    load_path = os.path.join(root_path,'Saved_files','trained_model',os.path.basename(__file__)[:-3],'train_oneline_({})'.format(params['net_name']),data_info['dataset'],'Seed_{}'.format(params["RANDOM_SEED"]),'sub_{}'.format(sub))

    if not os.path.exists(load_path):
        print('Error,model does not exist!')
    net.load_state_dict(torch.load(os.path.join(load_path,'model.pth'),map_location= device),strict= True)
    print('Model loaded successfully!')

    net.trans_mode()
    test_accus = []

    for _ in range(1):
        few_shot_x,few_shot_y, test_dataset = get_few_shot_test(sub,data_info['data_path'],5)
        test_accus.append(get_online_test_info(few_shot_x,few_shot_y,net,test_dataset,device = device)[1])
    test_accu = sum(test_accus)/len(test_accus)

    return test_accu


#2a
data_info = {
    'dataset':'BCIC_2a',
    'sub_num':9,
    'class_num':4,
    'domain_num':8,
    'channels':22,
    'data_path':os.path.join(root_path,'Data','BCIC_2a_0_4_0_38HZ'),
}

#HGD
# data_info = {
#     'dataset':'HGD_0_125HZ',
#     'sub_num':14,
#     'class_num':4,
#     'domain_num':13,
#     'channels':44,
#     'data_path':os.path.join(root_path,'Data','HGD_0_125HZ'),
# }

def run(sub, pparams,params,pro_device):
        
        set_seed(params["RANDOM_SEED"])
        pid = os.getpid()
        print('sub {} start! pid:{} device:{}'.format(sub,pid,pro_device[pid]))
        
        pparams["device"] = pro_device[pid]
        params["device"]= pro_device[pid]

        #pretraining
        accu = pre_train(sub,pparams)

        #trainning
        accu = train_online(sub,params)  
                
        #test
        accu = test(sub,params)

        print('sub {}:{}'.format(sub,accu))
        return {"sub":sub, "accu":accu, "params":params}


def callback(res):
        print('<Process%s> subject %s accu %s' %(os.getpid(),res['sub'], str(res["accu"])))


if __name__ == '__main__':

    if not os.path.exists(data_info['data_path']):
        print('{} does not exist!'.format(data_info['data_path']))
        from Data_process.process_function import Load_BCIC_2a_raw_data
        Load_BCIC_2a_raw_data()

    # The configure of pre-training
    pretrain_params = {
  
        'net_name':'CS_EEGPNet',
        'pretrain_name':'CS_EEGPNet',
        

        'channels':data_info['channels'],
        'batch_norm':FSFN2d, # nn.BatchNorm2d,
        'class_num':data_info['class_num'],

        'device': "cpu", #"cuda",


        'lr':0.001,                   
        'epoch':80,                  
        'dead_line':24,               
        'k_shot':8,                   
        'spatial_filter_number':30,   
        'feature_channels': 40,       
        'base_drop_out':0.5,        
  
    }

    # The configure of training
    params = copy.deepcopy(pretrain_params)
 
    params['batch_norm'] = FSFN2d
    params['lr']         = 0.0001                    
    params['epoch']      = 60                   
    params['dead_line']  = 24              
    params['k_shot']     = 1                 

    if pretrain_params['device'] == 'cuda':
        ctx = torch.multiprocessing.get_context("spawn")
        pool = ctx.Pool(3)

        device_pool = [1,2,0]
        pro_device = {}
        for i,p in enumerate(pool._pool):
            pro_device[p.pid] =  'cuda:{}'.format(device_pool[i%len(device_pool)])
        print(pro_device)

        all_accu = []
        for sub in range(1,data_info['sub_num']+1): 
            for random_seed in range(30,33):
                pparams = copy.deepcopy(pretrain_params)
                pparams["RANDOM_SEED"]= random_seed

                params = copy.deepcopy(params)
                params["RANDOM_SEED"]= random_seed

                accu = pool.apply_async(run,args=(sub, pparams,params,pro_device),callback=callback)
                all_accu.append(accu)

        pool.close()
        pool.join()
        res_l = all_accu
        res_l = [res.get() for res in res_l]

        print([res['accu'] for res in res_l])
        all_accu = np.array([res['accu'] for res in res_l]).mean()
        print(all_accu)

    else:
        all_accu = []
        for sub in range(1,data_info['sub_num']+1): 
            for random_seed in range(30,35):
                pparams = copy.deepcopy(pretrain_params)
                pparams["RANDOM_SEED"]= random_seed

                params = copy.deepcopy(params)
                params["RANDOM_SEED"]= random_seed

                #pretraining
                accu = pre_train(sub,pparams)

                #trainning
                accu = train_online(sub,params)  
                
                #test
                accu = test(sub,params)

                print('sub {}--seed {}-- accu {}'.format(sub,random_seed,accu))
                all_accu.append(accu)


    