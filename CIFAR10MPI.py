#import mpi4py
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import random
import os 
from model import trainClient
import copy
import torch
#from callClient import continueTrain
#Open communication()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(str(rank) + ": online")
serverWeight = []
if(rank == 0):
    #initialize server variables
    serverFlag = 0
    size = comm.Get_size()
    print(size)
else:
    clientFlag = 0
    print(str(rank) + "Client Waving: Starting the train")    
    net = trainClient(copyFlag=0,serverWeights=1)
    #communicate weights to server
    netcopy = copy.deepcopy(net)
    clientPayload = {"Weight": netcopy.state_dict(), "demand": 5, "supply" : 5}
    #comm.send(net.state_dict(), dest=0, tag=11)
    comm.send(clientPayload, dest=0)
    #initialize client variables initialize client variables   
if(rank == 0):
#prints the number of nodes (#clients + 1 server)# #prints the number of nodes (#clients + 1
#server)
    payload = []
    for i in range(1, 3, 1):
        payload.append(comm.recv(source=i))
    #print(payload)    
    serverWeight = payload[0]["Weight"]
    for i in range(len(payload)):
        for key in payload[0]["Weight"]:
            serverWeight[key] = (serverWeight[key] + payload[i]["Weight"][key]) 
    #print(serverWeight)
    for i in range(len(payload)):
        for key in payload[0]["Weight"]:
            serverWeight[key] = torch.div(serverWeight[key], 2)
serverWeight = comm.bcast(serverWeight, root=0) 

for rounds in range(1, 5, 1):    
    if(rank > 0):
        clientFlag = 0
        print(str(rank) + "Client Waving: Starting the train")    
        net = trainClient(copyFlag=1,serverWeights=serverWeight)
        #communicate weights to server
        netcopy = copy.deepcopy(net)
        clientPayload = {"Weight": netcopy.state_dict(), "demand": 5, "supply" : 5}
        #comm.send(net.state_dict(), dest=0, tag=11)
        comm.send(clientPayload, dest=0)               
    if(rank == 0):    
        payload = []
        for i in range(1, 3, 1):
            payload.append(comm.recv(source=i))
            print(payload)    
            serverWeight = payload[0]["Weight"]
            for i in range(len(payload)):
                for key in payload[0]["Weight"]:
                    serverWeight[key] = (serverWeight[key] + payload[i]["Weight"][key]) 
            #print(serverWeight)
            for i in range(len(payload)):
                for key in payload[0]["Weight"]:
                    serverWeight[key] = torch.div(serverWeight[key], 2)
    serverWeight = comm.bcast(serverWeight, root=0) 
    print("Rounds:")
    print(rounds)    
    #serverWeight = []
    #for k in range(len(payload)):
    #    serverWeight[0] += payload[k]
    #lambda x: x["Weight"]
    #print(np.average([x(i) for i in payload]))


