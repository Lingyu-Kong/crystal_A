import sys
sys.path.append("..") 
import time
import wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F
from memory.memory import Memory_Fixed
from critic.value_critic import Value_Critic
import config.config as config
from utils.tensor_utils import to_tensor

"""
Here we randomly generated some substructures and use rollout mean as the groundtruth.
"""

wandb.login()
wandb.init(project="Substructure_Critic", entity="kly20")

memory_params=config.memory_params
value_critic_params=config.value_critic_params
data_paths=config.data_paths
NUM_EPOCHS=10000
BATCH_SIZE=32


if __name__=="__main__":
    memory=Memory_Fixed(**memory_params)
    value_critic=Value_Critic(**value_critic_params)
    memory.memory_init(data_paths)
    # ## visualize ground truths 
    # values=memory.ucb_memory.tolist()
    # print(values)
    # plt.figure()
    # plt.plot(values,label="ground truth")
    # plt.legend()
    # wandb.log({"ground truth":plt})
    for epoch in range(NUM_EPOCHS):
        start_time=time.time()
        batch_structures,batch_sizes,batch_values=memory.sample(BATCH_SIZE,if_softmax=False,if_train=True)
        batch_structures=batch_structures[:,0:int(batch_sizes[0]),:]
        predicted_values=value_critic(to_tensor(batch_structures)).squeeze(-1)
        loss=F.mse_loss(predicted_values,to_tensor(batch_values))
        value_critic.optimizer.zero_grad()
        loss.backward()
        value_critic.optimizer.step()
        value_critic.scheduler.step()
        end_time=time.time()
        wandb.log({"train_loss":loss.item()})
        if epoch%100==0:
            print("epoch: {}, loss: {}, time: {}".format(epoch,loss.item(),end_time-start_time))
    # value_critic.save_model()
    ## test the model
    for i in range(5):
        batch_structures,batch_sizes,batch_values=memory.sample(BATCH_SIZE,if_softmax=False,if_train=False)
        batch_structures=batch_structures[:,0:int(batch_sizes[0]),:]
        predicted_values=value_critic(to_tensor(batch_structures)).squeeze(-1)
        loss=F.mse_loss(predicted_values,to_tensor(batch_values))
        wandb.log({"test_loss":loss.item()})

    ## visualize predicted values
    ground_truth=[]
    predicted=[]
    for i in range(memory.memory_size):
        substructure=memory.substructure_memory[i]
        ground_truth.append(memory.ucb_memory[i])
        predicted.append(value_critic(to_tensor(substructure).unsqueeze(-1)).squeeze(-1).item())
    plt.figure()
    plt.plot(ground_truth,label="ground truth")
    plt.plot(predicted,label="predicted")
    plt.legend()
    wandb.log({"performance":plt})
