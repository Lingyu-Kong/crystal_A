import torch
import torch.nn as nn

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

shared_params={
    "num_atoms":60,
    "device":device,
}

substructure_gen_params={
    "num_atoms":shared_params["num_atoms"],
    "pos_scale":3.0,
    "threshold":0.8,
    "max_steps":1000,
    "num_substructures":5000,
}

memory_params={
    "memory_size":2000,
    "num_atoms":shared_params["num_atoms"],
}

gnn_params = {
    "device":device,
    "num_atoms":shared_params["num_atoms"],
    "mlp_hidden_size":256,
    "mlp_layers":2,
    "latent_size":64,
    "use_layer_norm":False,
    "num_message_passing_steps":6,
    "global_reducer":"sum",
    "node_reducer":"sum",
    "dropedge_rate":0.1,
    "dropnode_rate":0.1,
    "dropout":0.1,
    "layernorm_before":False,
    "use_bn":False,
    "cycle":1,
    "node_attn":True,
    "global_attn":True,
}

mlp_params = {
    "input_size":gnn_params["latent_size"],
    "output_sizes":[gnn_params["latent_size"]]*2+[1],
    "use_layer_norm":False,
    "activation":nn.ReLU,
    "dropout":0.1,
    "layernorm_before":False,
    "use_bn":False,
}

value_critic_params={
        "gnn_params":gnn_params,
        "mlp_params":mlp_params,
        "lr":5e-4,
        "decay_interval":200,
        "decay_rate":0.95,
        "device":shared_params["device"],
}

data_paths=["/substructure/substructures_"+str(shared_params["num_atoms"])+".npy"
            ,"/substructure/substructure_sizes_"+str(shared_params["num_atoms"])+".npy"]