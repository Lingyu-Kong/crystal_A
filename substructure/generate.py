import sys
sys.path.append("..") 
import numpy as np
from utils.bfgs_utils import relax
import config.config as config

"""
We generate new substructure by cutting local minimal into two parts with a random 3D plane
"""

def single_conform_sample(num_atoms,pos_scale,threshold):
    pos=np.zeros((num_atoms,3))
    for i in range(num_atoms):
        if_continue=True
        while if_continue:
            new_pos=np.random.rand(3)*2*pos_scale-pos_scale
            if_continue=False
            for j in range(i):
                distance=np.linalg.norm(new_pos-pos[j],ord=2)
                if distance<threshold:
                    if_continue=True
                    break
        pos[i,:]=new_pos
    return pos

class Generator(object):
    def __init__(
        self,
        num_atoms:int,
        pos_scale:float,
        threshold:float,
        max_steps:int,
        num_substructures:int,
    ):
        self.num_atoms=num_atoms
        self.pos_scale=pos_scale
        self.threshold=threshold
        self.max_steps=max_steps
        self.num_substructures=num_substructures

    def generate(self):
        substructures=np.zeros((self.num_substructures,self.num_atoms,3))
        substructure_sizes=np.zeros(self.num_substructures,dtype=int)
        for i in range(self.num_substructures):
            conform=single_conform_sample(self.num_atoms,self.pos_scale,self.threshold)
            _,_,conform=relax(conform,self.max_steps)
            choices=np.random.choice(self.num_atoms,self.num_atoms//2,replace=False)
            substructures[i,np.arange(self.num_atoms//2),:]=conform[choices,:]
            substructure_sizes[i]=self.num_atoms//2
            if i%100==0:
                print("{}/{}".format(i+1,self.num_substructures))
        ## save to file
        np.save("substructures_"+str(self.num_atoms)+".npy",substructures)
        np.save("substructure_sizes_"+str(self.num_atoms)+".npy",substructure_sizes)

if __name__=="__main__":
    substructure_gen_params=config.substructure_gen_params
    generator=Generator(**substructure_gen_params)
    generator.generate()
    print(str(generator.num_atoms)+" atoms' substructures generated")