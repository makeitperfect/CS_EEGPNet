import torch
import os
import sys
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)
from utils import *
root_path = os.path.split(current_path)[0]
sys.path.append(root_path)
from network.utils import *

'''
========
BCIC 2a
========
1-shot
--------
seed 0  seed 1  seed 2  seed 3  seed 4
--------------------------------------
0.7332	0.7390	0.7386	0.7203	0.7031
0.5469	0.5430	0.5163	0.5444	0.5013
0.7921	0.8267	0.8336	0.8172	0.8288
0.5715	0.5708	0.5885	0.5497	0.5516
0.6808	0.6831	0.6378	0.6753	0.6629
0.5986	0.5963	0.5844	0.5582	0.6055
0.8065	0.7813	0.7710	0.7865	0.7509
0.8024	0.8034	0.7769	0.7822	0.7926
0.7164	0.7470	0.7346	0.7620	0.7478
0.69024473

2-shot
--------
seed 0  seed 1  seed 2  seed 3  seed 4
--------------------------------------
0.7347	0.7467	0.7543	0.7302	0.7206
0.5651	0.5572	0.5295	0.5526	0.5186
0.8120	0.8336	0.8561	0.8249	0.8398
0.5792	0.5806	0.5968	0.5487	0.5599
0.7046	0.6956	0.6550	0.6876	0.6690
0.6177	0.5936	0.6072	0.5788	0.6164
0.8093	0.7876	0.7777	0.7966	0.7586
0.8129	0.8152	0.7996	0.8045	0.8057
0.7286	0.7558	0.7580	0.7743	0.7804
0.7029245

3-shot
--------
seed 0  seed 1  seed 2  seed 3  seed 4
--------------------------------------
0.7459	0.7499	0.7566	0.7301	0.7259
0.5710	0.5575	0.5317	0.5526	0.5324
0.8176	0.8361	0.8570	0.8406	0.8462
0.5902	0.5893	0.6072	0.5715	0.5654
0.7098	0.6999	0.6597	0.6960	0.6756
0.6231	0.5996	0.6170	0.5797	0.6186
0.8145	0.7907	0.7793	0.7997	0.7650
0.8167	0.8202	0.8057	0.8098	0.8077
0.7384	0.7668	0.7637	0.7756	0.7886
0.7088022

4-shot
--------
seed 0  seed 1  seed 2  seed 3  seed 4
--------------------------------------
0.7534	0.7562	0.7594	0.7338	0.7246
0.5720	0.5630	0.5375	0.5554	0.5371
0.8283	0.8413	0.8513	0.8454	0.8505
0.5955	0.5960	0.6129	0.5733	0.5672
0.7130	0.6961	0.6552	0.6987	0.6808
0.6275	0.6073	0.6195	0.5827	0.6205
0.8167	0.7946	0.7862	0.8029	0.7677
0.8200	0.8237	0.8086	0.8143	0.8133
0.7368	0.7701	0.7733	0.7788	0.7895
0.71226585

5-shot
--------
seed 0  seed 1  seed 2  seed 3  seed 4
--------------------------------------
0.7585	0.7595	0.7635	0.7359	0.7288
0.5757	0.5677	0.5404	0.5600	0.5458
0.8338	0.8462	0.8549	0.8471	0.8522
0.5990	0.5952	0.6145	0.5837	0.5688
0.7138	0.6955	0.6652	0.7022	0.6813
0.6325	0.6069	0.6245	0.5792	0.6232
0.8210	0.7976	0.7895	0.8079	0.7653
0.8200	0.8257	0.8110	0.8167	0.8165
0.7409	0.7692	0.7750	0.7855	0.7889
0.7152539


20-shot
--------
seed 0  seed 1  seed 2  seed 3  seed 4
--------------------------------------
0.7598	0.7697	0.7716	0.7402	0.7372
0.5763	0.5737	0.5490	0.5633	0.5596
0.8396	0.8469	0.8582	0.8559	0.8546
0.6028	0.6136	0.6227	0.5856	0.5738
0.7224	0.7002	0.6746	0.7019	0.6869
0.6406	0.6155	0.6285	0.5811	0.6313
0.8166	0.8031	0.7951	0.8153	0.7742
0.8229	0.8312	0.8147	0.8215	0.8229
0.7441	0.7753	0.7869	0.7916	0.7984
0.7211318

'''



class Block(nn.Module):
    def __init__(self,in_channel,out_channel = None,skip  = True, bn = None):
        super(Block,self).__init__()

        bn = nn.BatchNorm2d if bn is None else bn
        out_channel = in_channel if out_channel== None else out_channel

        self.act = nn.ELU()
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size = 1)
        self.bn1 = bn(out_channel)

        self.conv2 = nn.Conv2d(out_channel,out_channel,kernel_size = (1,25),dilation = 1,padding = 'same',groups= out_channel,bias= False)
        self.bn2 = bn(out_channel)
        
        self.skip = None if skip is False else Conv2dWithConstraint(in_channel,out_channel,kernel_size = 1,max_norm= 0.25)
        
        self.bn3 = bn(out_channel)
        self.drop = nn.Dropout(0.5)

        self.bn4 = bn(out_channel)


    def online(self):
        self.bn1.online()
        self.bn2.online()
        self.bn3.online()
        self.bn4.online()

    def offline(self):
        self.bn1.offline()
        self.bn2.offline()
        self.bn3.offline()
        self.bn4.offline()

    def forward(self,x):
        skip = x
        h = self.bn1(self.conv1(x))
        h = self.bn2(self.conv2(h))

        if self.skip is not None:
            h = self.bn4(self.drop(self.bn3(self.skip(x)))+h)
 
        return self.act(h)

class CS_EEGPNet(nn.Module):   
  
    def __init__(self, *args, **kwargs):
        super(CS_EEGPNet, self).__init__()

        self.spatial_filter_number  = kwargs['spatial_filter_number']
        self.feature_channels = kwargs['feature_channels']
        self.base_drop_out = kwargs['base_drop_out']
        self.channels = kwargs['channels']
        self.batch_norm = kwargs['batch_norm'] # Batch normlization or Few-shot feature normalization.
        self.class_num = kwargs['class_num']

        self.first_order_feature = nn.Sequential(
            #EEG Mobile Block
            Conv2dWithConstraint(1,self.spatial_filter_number,kernel_size=(self.channels,1)),
            self.batch_norm(self.spatial_filter_number),#1
            CRG_Block(self.spatial_filter_number,2,15,55,125),
            self.batch_norm(self.spatial_filter_number*6),#3
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,3),stride=3),
            nn.Dropout(self.base_drop_out),

            #EEG Incption Block 1
            Block(self.spatial_filter_number*6,self.spatial_filter_number*6,bn = self.batch_norm,skip = True),#7
            nn.AvgPool2d(kernel_size=(1,3),stride=3), 
            nn.Dropout(self.base_drop_out),

            #EEG Incption Block 2
            Block(self.spatial_filter_number*6,self.feature_channels,bn= self.batch_norm,skip = True),#10
            nn.AvgPool2d(kernel_size=(1,3),stride=3),
            nn.Dropout(self.base_drop_out),
        )

        # FC
        self.adap_layer = nn.Sequential(
            LinearWithConstraint(self.feature_channels*37,64, max_norm=0.1),
            nn.Dropout(0.5),
        )

        self.pub_proto = nn.Parameter(torch.zeros(self.class_num,64,dtype = torch.float))
        torch.nn.init.xavier_uniform_(self.pub_proto)

        


        self.mode = 0 #0,train_base,1

        self.offline_mode = True

    def base_mode(self):
        self.mode = 0
        self.adap_layer[0].maxnorm = 0.5

    def trans_mode(self):
        self.mode = 1
        self.adap_layer[0].maxnorm = 0.1

    def online(self):
        self.offline_mode = False
        norm_index = [1,3,7,10]
        for i in norm_index:
            self.first_order_feature[i].online()
    
    def offline(self):
        self.offline_mode = True
        norm_index = [1,3,7,10]
        for i in norm_index:
            self.first_order_feature[i].offline()

    def base_forward(self,feature):

        feature = feature[:,:,:1000]
        B,C,S = feature.shape
        h = feature.reshape(B,1,C,S)

        h1 = self.first_order_feature(h)

        h = torch.flatten(h1,1)

        # h = self.adap_layer[:-1](h)
        h = self.adap_layer(h)
        # h = self.mi_calssifier(h)
        
        distance = pairwise_distances(h,self.pub_proto,'cosine')

        return distance

    def proto_forward(self,support_x,support_y,query_x):
        # print(support_y)
        support_x = support_x[:,:,:1000]
        query_x  =  query_x[:,:,:1000]
        # B,C,S = support_x.shape
        h0 = support_x[:,None]
        B,C,S = query_x.shape
        h1 = query_x[:,None]
        
        self.offline()
        h0 = self.first_order_feature(h0)[:,:,0]

        self.online()
        h1 = self.first_order_feature(h1)[:,:,0]
        
        h0 = torch.flatten(h0,1)
        h1 = torch.flatten(h1,1)

        h0 = self.adap_layer(h0)
        h1 = self.adap_layer(h1)


        h0 = rearrange(h0,'(a b) c -> a b c',a = self.class_num)
        
        h0 = torch.cat([h0,self.pub_proto[:,None]],dim = 1).mean(dim = 1)

        # distance = pairwise_distances(h1,h0,'dot')
        distance = pairwise_distances(h1,h0,'cosine')
        # distance = pairwise_distances(h1,h0,'l2')
    
        return distance

    def forward(self,support_x,support_y = None,query_x = None):
        if self.mode == 0:
            return self.base_forward(support_x)

        return self.proto_forward(support_x,support_y,query_x)
 











