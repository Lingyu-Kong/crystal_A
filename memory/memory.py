import numpy as np
import time
from torch.nn import Softmax
from utils.tensor_utils import to_tensor
from utils.bfgs_utils import relax

"""
store substructure and ucb-value
"""

class Memory(object):
    def __init__(
        self,
        memory_size:int,
        num_atoms:int,
    ):
        self.substructure_memory=np.zeros((memory_size, num_atoms,3))
        self.substructure_size=np.zeros(memory_size)
        self.ucb_memory=np.zeros(memory_size)
        self.memory_size=memory_size
        self.memory_top=0
        self.num_atoms=num_atoms

    def memory_init(self,path=None):
        if path is not None and len(path)==2:
            substructure_memory=np.load(path[0])
            substructure_size=np.load(path[1])
            data_length=substructure_memory.shape[0]
            if data_length>self.memory_size:
                choices=np.random.choice(data_length,size=self.memory_size,replace=False)
                self.substructure_memory=substructure_memory[choices]
                self.substructure_size=substructure_size[choices]
                self.memory_top=self.memory_size+1
            else:
                self.substructure_memory[0:data_length]=substructure_memory
                self.substructure_size[0:data_length]=substructure_size
                self.memory_top=data_length+1
            for i in range(min(self.memory_top,self.memory_size)):
                choices=np.random.choice(min(self.memory_top,self.memory_size),size=10)
                start_time=time.time()
                count=0
                for j in range(10):
                    substructure_1=self.substructure_memory[i][0:self.substructure_size[i]]
                    substructure_2=self.substructure_memory[choices[j]][0:self.substructure_size[choices[j]]]
                    conform=np.concatenate((substructure_1,substructure_2),axis=0)
                    _,energy,_=relax(conform)
                    if not np.isnan(energy):
                        self.ucb_memory[i]+=-energy
                        count+=1
                    if count>=5:
                        break
                self.ucb_memory[i]=self.ucb_memory[i]/count
                end_time=time.time()
                print("{} done time: {}, value: {}".format(i,end_time-start_time,self.ucb_memory[i]))
            mean_ucb=np.mean(self.ucb_memory)
            self.ucb_memory=self.ucb_memory-mean_ucb
            ## save
            np.save("substructures_"+str(self.num_atoms)+".npy",self.substructure_memory)
            np.save("substructure_sizes_"+str(self.num_atoms)+".npy",self.substructure_size)
            np.save("value_"+str(self.num_atoms)+".npy",self.ucb_memory)
        elif path is not None and len(path)==3:
            substructure_memory=np.load(path[0])
            substructure_size=np.load(path[1])
            data_length=substructure_memory.shape[0]
            if data_length>self.memory_size:
                choices=np.random.choice(data_length,size=self.memory_size,replace=False)
                self.substructure_memory=substructure_memory[choices]
                self.substructure_size=substructure_size[choices]
                self.ucb_memory=np.load(path[2])[choices]
                self.memory_top=self.memory_size+1
            else:
                self.substructure_memory[0:data_length]=substructure_memory
                self.substructure_size[0:data_length]=substructure_size
                self.ucb_memory[0:data_length]=np.load(path[2])
                self.memory_top=data_length+1


    def sample(self,batch_size=2,if_softmax=True):
        if if_softmax:
            probs=Softmax(dim=0)(10*to_tensor(self.ucb_memory)).detach().cpu().numpy()
            choices=np.random.choice(min(self.memory_top,self.memory_size),size=batch_size,p=probs)
        else:
            choices=np.random.choice(min(self.memory_top,self.memory_size),size=batch_size)
        substructures=self.substructure_memory[choices]
        substructure_sizes=self.substructure_size[choices]
        ucb_values=self.ucb_memory[choices]
        return substructures,substructure_sizes,ucb_values

    def newin_update(self,substurecture,substructure_size,ucb_value):
        min_index=np.argmin(self.ucb_memory)
        if ucb_value>self.ucb_memory[min_index]:
            self.substructure_memory[min_index]=substurecture
            self.substructure_size[min_index]=substructure_size
            self.ucb_memory[min_index]=ucb_value


class Memory_Fixed(object):
    def __init__(
        self,
        memory_size:int,
        num_atoms:int,
        train_test_split:float=0.8,
    ):
        self.substructure_memory=np.zeros((memory_size, num_atoms,3))
        self.substructure_size=np.zeros(memory_size)
        self.ucb_memory=np.zeros(memory_size)
        self.train_size=int(memory_size*train_test_split)
        self.test_size=int(memory_size-self.train_size)
        self.memory_size=memory_size
        self.num_atoms=num_atoms

    def memory_init(self,path=None):
        if path is not None and len(path)==2:
            substructure_memory=np.load(path[0])
            substructure_size=np.load(path[1])
            data_length=substructure_memory.shape[0]
            if data_length>self.memory_size:
                choices=np.random.choice(data_length,size=self.memory_size,replace=False)
                self.substructure_memory=substructure_memory[choices]
                self.substructure_size=substructure_size[choices]
            else:
                self.substructure_memory[0:data_length]=substructure_memory
                self.substructure_size[0:data_length]=substructure_size
            for i in range(self.memory_size):
                choices=np.random.choice(self.memory_size,size=10)
                start_time=time.time()
                count=0
                for j in range(10):
                    substructure_1=self.substructure_memory[i][0:self.substructure_size[i]]
                    substructure_2=self.substructure_memory[choices[j]][0:self.substructure_size[choices[j]]]
                    conform=np.concatenate((substructure_1,substructure_2),axis=0)
                    _,energy,_=relax(conform)
                    if not np.isnan(energy):
                        self.ucb_memory[i]+=-energy
                        count+=1
                    if count>=5:
                        break
                self.ucb_memory[i]=self.ucb_memory[i]/count
                end_time=time.time()
                print("{} done time: {}, value: {}".format(i,end_time-start_time,self.ucb_memory[i]))
            mean_ucb=np.mean(self.ucb_memory)
            self.ucb_memory=self.ucb_memory-mean_ucb
            ## save
            np.save("substructures_"+str(self.num_atoms)+".npy",self.substructure_memory)
            np.save("substructure_sizes_"+str(self.num_atoms)+".npy",self.substructure_size)
            np.save("value_"+str(self.num_atoms)+".npy",self.ucb_memory)
        elif path is not None and len(path)==3:
            substructure_memory=np.load(path[0])
            substructure_size=np.load(path[1])
            data_length=substructure_memory.shape[0]
            if data_length>self.memory_size:
                choices=np.random.choice(data_length,size=self.memory_size,replace=False)
                self.substructure_memory=substructure_memory[choices]
                self.substructure_size=substructure_size[choices]
                self.ucb_memory=np.load(path[2])[choices]
            else:
                self.substructure_memory[0:data_length]=substructure_memory
                self.substructure_size[0:data_length]=substructure_size
                self.ucb_memory[0:data_length]=np.load(path[2])


    def sample(self,batch_size=2,if_softmax=True,if_train=True):
        if if_softmax:
            probs=Softmax(dim=0)(10*to_tensor(self.ucb_memory)).detach().cpu().numpy()
            if if_train:
                choices=np.random.choice(self.train_size,size=batch_size,p=probs)
            else:
                choices=np.random.choice(self.test_size,size=batch_size,p=probs)
                choices+=self.train_size
        else:
            if if_train:
                choices=np.random.choice(self.train_size,size=batch_size)
            else:
                choices=np.random.choice(self.test_size,size=batch_size)
                choices+=self.train_size
        substructures=self.substructure_memory[choices]
        substructure_sizes=self.substructure_size[choices]
        ucb_values=self.ucb_memory[choices]
        return substructures,substructure_sizes,ucb_values

    def newin_update(self,substurecture,substructure_size,ucb_value):
        min_index=np.argmin(self.ucb_memory)
        if ucb_value>self.ucb_memory[min_index]:
            self.substructure_memory[min_index]=substurecture
            self.substructure_size[min_index]=substructure_size
            self.ucb_memory[min_index]=ucb_value