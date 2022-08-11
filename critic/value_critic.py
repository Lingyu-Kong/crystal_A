import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from gnn.ignn import IGNN
from gnn.metalayer import MLPwoLastAct

class Value_Critic(nn.Module):
    def __init__(
        self,
        gnn_params:dict,
        mlp_params:dict,
        lr:float,
        decay_interval:int,
        decay_rate:float,
        device:torch.device,
    ):
        super(Value_Critic, self).__init__()
        self.gnn=IGNN(**gnn_params)
        self.final_mlp=MLPwoLastAct(**mlp_params)
        self.device=device
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        self.scheduler = StepLR(self.optimizer, decay_interval, decay_rate)
        self.gnn.to(device)
        self.final_mlp.to(device)
        self.weight_init(self.gnn)
        self.weight_init(self.final_mlp)
    
    def forward(self,substructures):
        """
        substructures:[batch_size,sub_size,3]
        """
        _,_,global_attr=self.gnn(substructures)
        values=self.final_mlp.forward(global_attr)
        return values

    def save_model(self,path):
        torch.save(self.state_dict(),path)
    
    def load_model(self,path):
        self.load_state_dict(torch.load(path))

    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
