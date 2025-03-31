import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from Filters.EKF_test import EKFTest

from Simulations.Ctrl_Extended_sysmdl import ControlSystemModel
from Simulations.utils import DataGen,Short_Traj_Split
import Simulations.config as config

from Pipelines.Pipeline_EKF import Pipeline_EKF
from Pipelines.Pipeline_Recovery import Pipeline_Recovery_Controller

from datetime import datetime

from Networks.KalmanNet_nn import KalmanNetNN
from Networks.RecoveryController import RecoveryController
from Networks.RecoveryNetwork import RecoveryNetwork

from Simulations.Hillclimbing_Car.parameters import m1x_0, m2x_0, m, n, p, pid_params, dt, \
    f, h, Q_structure, R_structure, init_setpoint, setpoint_change

# from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n,\
# f, h, hRotate, H_Rotate, H_Rotate_inv, Q_structure, R_structure

print("Pipeline Start")
################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

###################
###  Settings   ###
###################
args = config.general_settings()
### dataset parameters
args.N_E = 1000
args.N_CV = 100
args.N_T = 200
args.T = 100
args.T_test = 100
### training parameters
args.use_cuda = False # use GPU or not
args.n_steps = 2000
args.n_batch = 30
args.lr = 1e-3
args.wd = 1e-3

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

offset = 0 # offset for the data
chop = False # whether to chop data sequences into shorter sequences
path_results = 'ModelWeights/'
DatafolderName = 'Simulations/Hillclimbing_Car/data' + '/'
switch = 'partial' # 'full' or 'partial' or 'estH'
   
# noise q and r
r2 = torch.tensor([0.1]) # [100, 10, 1, 0.1, 0.01]
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)

Q = q2[0] * Q_structure
R = r2[0] * R_structure

print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))

traj_resultName = ['traj_hillcar_rq1030_T100.pt']
dataFileName = ['data_hillcar_rq1030_T100.pt']

#########################################
###  Generate and load data DT case   ###
#########################################

sys_model = ControlSystemModel(f, Q, h, R, args.T, args.T_test, m, n, p, pid_params, dt, init_setpoint, setpoint_change)# parameters for GT
sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0

print("Start Data Gen")
DataGen(args, sys_model, DatafolderName + dataFileName[0])
print("Data Load")
print(dataFileName[0])
[train_input_long, train_target_long, train_setpoint_long, 
 cv_input, cv_target, cv_setpoint, 
 test_input, test_target, test_setpoint,_,_,_] =  torch.load(DatafolderName + dataFileName[0], map_location=device)  
if chop: 
   print("chop training data")    
   [train_target, train_input, train_init] = Short_Traj_Split(train_target_long, train_input_long, args.T)
   # [cv_target, cv_input] = Short_Traj_Split(cv_target, cv_input, args.T)
else:
   print("no chopping") 
   train_target = train_target_long[:,:,0:args.T]
   train_input = train_input_long[:,:,0:args.T] 
   # cv_target = cv_target[:,:,0:args.T]
   # cv_input = cv_input[:,:,0:args.T]  

print("trainset size:",train_target.size())
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())


########################
### Evaluate Filters ###
########################
### Evaluate EKF true
# print("Evaluate EKF true")
# [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(args, sys_model, test_input, test_target)
# ### Evaluate EKF partial
# print("Evaluate EKF partial")
# [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(args, sys_model_partial, test_input, test_target)

# ### Save trajectories
# trajfolderName = 'Filters' + '/'
# DataResultName = traj_resultName[0]
# EKF_sample = torch.reshape(EKF_out[0],[1,m,args.T_test])
# target_sample = torch.reshape(test_target[0,:,:],[1,m,args.T_test])
# input_sample = torch.reshape(test_input[0,:,:],[1,n,args.T_test])
# torch.save({
#             'EKF': EKF_sample,
#             'ground_truth': target_sample,
#             'observation': input_sample,
#             }, trajfolderName+DataResultName)

#####################
### Evaluate KNet ###
#####################

## KNet with full info ####################################################################################
################
## KNet full ###
################  
## Build Neural Network
print("KNet with full model info")
KNet_model = KalmanNetNN()
KNet_model.NNBuild(sys_model, args)
controller = RecoveryController(
   input_size = m,
   hidden_size=100,
   num_layers=1,
   out_size=1,
   clip_output=True
)
recovery_network = RecoveryNetwork(KNet_model, controller)
# ## Train Neural Network
KNet_Pipeline = Pipeline_Recovery_Controller(strTime, "KNet", "KNet")
KNet_Pipeline.setssModel(sys_model)
KNet_Pipeline.setModel(recovery_network)
print("Number of trainable parameters for KNet:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
KNet_Pipeline.setTrainingParams(args) 

if(chop):
    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch, min_value_train, max_value_train] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results,randomInit=True,train_init=train_init)
else:
    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch, min_value_train, max_value_train] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
## Test Neural Network
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,Knet_out,RunTime] = KNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results, min_value_train, max_value_train)

####################################################################################
print("Pipeline Done.")
   





