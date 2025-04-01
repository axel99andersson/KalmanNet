import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from .Extended_sysmdl import SystemModel
from Controllers.PID import PIDController
from Networks.RecoveryNetwork import RecoveryNetwork
from typing import List

class ControlSystemModel(SystemModel):
    def __init__(self, f, Q, h, R, T, T_test, m, n, p, pid_params, dt, init_setpoint, setpoint_change, prior_Q=None, prior_Sigma=None, prior_S=None):
        super().__init__(f, Q, h, R, T, T_test, m, n, prior_Q, prior_Sigma, prior_S)
        self.p = p      # Dimension of control signal
        self.pid_params = pid_params
        self.dt = dt
        self.init_setpoint = init_setpoint
        self.setpoint_change = setpoint_change

    def InitSequence(self, m1x_0, m2x_0):
        return super().InitSequence(m1x_0, m2x_0)
    
    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):
        return super().Init_batched_sequence(m1x_0_batch, m2x_0_batch)
    
    def UpdateCovariance_Matrix(self, Q, R):
        return super().UpdateCovariance_Matrix(Q, R)
    
    def GenerateSequence(self, Q_gen, R_gen, T):
        # Init PID controller
        self.pid_controller = PIDController(*self.pid_params, torch.zeros(1, self.n,1), self.dt)
        # Pre allocate an array for current state
        self.x = torch.zeros(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.zeros(size=[self.n, T])
        # Pre allocate an array for for current control signal
        self.u = torch.zeros(size=[self.p, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev
        
        ut = torch.zeros(self.p, 1)

        # Randomized mission: setpoints change over time
        setpoint = torch.ones(self.p, 1)*self.init_setpoint
        setpoint = setpoint + torch.rand(self.p, 1)*(self.init_setpoint/4) - torch.ones(self.p, 1)*(self.init_setpoint/8)
        for t in range(T):
            if t % (T // 4) == 0:  # Change setpoints at intervals
                setpoint = setpoint + torch.rand(self.p, 1)*self.setpoint_change - torch.ones(self.p, 1)*(self.setpoint_change/2)  # New target in [-1,1]
                self.pid_controller.set_setpoint(setpoint)
            
            # State evolution
            if torch.equal(Q_gen, torch.zeros(self.m, self.m)):
                xt = self.f(self.x_prev, ut)
            elif self.m == 1: # 1 dim noise
                xt = self.f(self.x_prev, ut)
                eq = torch.normal(mean=0, std=Q_gen)
                # Additive Process Noise
                xt = torch.add(xt,eq)
            else:
                xt = self.f(self.x_prev, ut)
                mean = torch.zeros([self.m])              
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                eq = distrib.rsample()
                eq = torch.reshape(eq[:], xt.size())
                # Additive Process Noise
                xt = torch.add(xt,eq)
            
            # Observation with noise
            yt = self.h(xt)
            if torch.equal(R_gen, torch.zeros(self.n, self.n)):
                yt = self.h(xt)
            if self.n == 1: # 1 dim noise
                er = torch.normal(mean=0, std=R_gen)
                # Additive Observation Noise
                yt = torch.add(yt,er)
            else:
                mean = torch.zeros([self.n])            
                distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
                er = distrib.rsample()
                er = torch.reshape(er[:], yt.size())       
                # Additive Observation Noise
                yt = torch.add(yt,er)
            
            # Compute control input using PID
            ut: torch.Tensor = self.pid_controller.compute_control(yt)
            
            # Save data
            self.x[:, t] = xt.squeeze()
            self.y[:, t] = yt.squeeze()
            self.u[:, t] = ut.squeeze()
            self.x_prev = xt


    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, args, size, T, randomInit=False):
        # Init PID controller
        self.pid_controller = PIDController(*self.pid_params, torch.zeros(size, self.n,1), self.dt)
        if(randomInit):
            # Allocate Empty Array for Random Initial Conditions
            self.m1x_0_rand = torch.zeros(size, self.m, 1)
            if args.distribution == 'uniform':
                ### if Uniform Distribution for random init
                for i in range(size):           
                    initConditions = torch.rand_like(self.m1x_0) * args.variance
                    self.m1x_0_rand[i,:,0:1] = initConditions.view(self.m,1)     
            
            elif args.distribution == 'normal':
                ### if Normal Distribution for random init
                for i in range(size):
                    distrib = MultivariateNormal(loc=torch.squeeze(self.m1x_0), covariance_matrix=self.m2x_0)
                    initConditions = distrib.rsample().view(self.m,1)
                    self.m1x_0_rand[i,:,0:1] = initConditions
            else:
                raise ValueError('args.distribution not supported!')
            
            self.Init_batched_sequence(self.m1x_0_rand, self.m2x_0)### for sequence generation
        else: # fixed init
            initConditions = self.m1x_0.view(1,self.m,1).expand(size,-1,-1)
            self.Init_batched_sequence(initConditions, self.m2x_0)### for sequence generation
    
        if(args.randomLength):
            # Allocate Array for Input and Target (use zero padding)
            self.Input = torch.zeros(size, self.n, args.T_max)
            self.States = torch.zeros(size, self.m, args.T_max)
            self.Target = torch.zeros(size, self.p, args.T_max)
            self.Setpoint = torch.zeros(size, self.p, args.T_max)
            self.lengthMask = torch.zeros((size,args.T_max), dtype=torch.bool)# init with all false
            # Init Sequence Lengths
            T_tensor = torch.round((args.T_max-args.T_min)*torch.rand(size)).int()+args.T_min # Uniform distribution [100,1000]
            for i in range(0, size):
                # Generate Sequence
                self.GenerateSequence(self.Q, self.R, T_tensor[i].item())
                # Training sequence input
                self.Input[i, :, 0:T_tensor[i].item()] = self.y             
                # States sequence                
                self.States[i, :, 0:T_tensor[i].item()] = self.x
                # Training sequence output
                self.Target[i, :, 0:T_tensor[i].item()] = self.u
                # Mask for sequence length
                self.lengthMask[i, 0:T_tensor[i].item()] = True

        else:
            # Allocate Empty Array for Input
            self.Input = torch.empty(size, self.n, T)
            # Allocate Empty Array for States
            self.States = torch.empty(size, self.m, T)
            # Allocate Empty Array for Target
            self.Target = torch.empty(size, self.p, T)
            # Allocate Empty Array for Setpoint
            self.Setpoint = torch.empty(size, self.p, T)
            # Set x0 to be x previous
            self.x_prev = self.m1x_0_batch
            xt = self.x_prev
            ut = torch.zeros(size, self.p, 1)
            setpoint = torch.ones(size, self.p, 1)*self.init_setpoint
            setpoint = setpoint +  torch.rand(size, self.p, 1)*(self.init_setpoint/4) - torch.ones(size, self.p, 1)*(self.init_setpoint/8)
            # Generate in a batched manner
            for t in range(0, T):
                ########################
                #### New Setpoint ######
                if t % (T // 4) == 0:  # Change setpoints at intervals
                    setpoint = setpoint + torch.rand(size, self.p, 1) * self.setpoint_change - torch.ones(size, self.p, 1)*(self.setpoint_change/2) # New target in [-1,1]
                    self.pid_controller.set_setpoint(setpoint)
                ########################
                #### State Evolution ###
                ########################

                   
                if torch.equal(self.Q,torch.zeros(self.m,self.m)):# No noise
                    xt = self.f(self.x_prev, ut)
                elif self.m == 1: # 1 dim noise
                    xt = self.f(self.x_prev, ut)
                    eq = torch.normal(mean=torch.zeros(size), std=self.Q).view(size,1,1)
                    # Additive Process Noise
                    xt = torch.add(xt,eq)
                else:            
                    xt = self.f(self.x_prev, ut)
                    mean = torch.zeros([size, self.m])              
                    distrib = MultivariateNormal(loc=mean, covariance_matrix=self.Q)
                    eq = distrib.rsample().view(size,self.m,1)
                    # Additive Process Noise
                    xt = torch.add(xt,eq)

                ################
                ### Emission ###
                ################
                # Observation Noise
                if torch.equal(self.R,torch.zeros(self.n,self.n)):# No noise
                    yt = self.h(xt)
                elif self.n == 1: # 1 dim noise
                    yt = self.h(xt)
                    er = torch.normal(mean=torch.zeros(size), std=self.R).view(size,1,1)
                    # Additive Observation Noise
                    yt = torch.add(yt,er)
                else:  
                    yt =  self.h(xt)
                    mean = torch.zeros([size,self.n])            
                    distrib = MultivariateNormal(loc=mean, covariance_matrix=self.R)
                    er = distrib.rsample().view(size,self.n,1)          
                    # Additive Observation Noise
                    yt = torch.add(yt,er)

                ut = self.pid_controller.compute_control(yt)

                ########################
                ### Squeeze to Array ###
                ########################
                # Save current Control signal to Trajectory Array
                self.Target[:, :, t] = torch.squeeze(ut,2)
                # Save Current State to Trajectory Array
                self.States[:, :, t] = torch.squeeze(xt,2)
                # Save Current Observation to Trajectory Array
                self.Input[:, :, t] = torch.squeeze(yt,2)
                # Save Current Setpoint to Trajectory Array
                self.Setpoint[:, :, t] = torch.squeeze(setpoint, 2)

                ################################
                ### Save Current to Previous ###
                ################################
                self.x_prev = xt
    
    def GenerateSequenceRecovery(self, Q_gen, R_gen, T, scaler_min, scaler_max, path_results, device='cpu', load_model=False, load_model_path=None):
        # Init PID controller
        self.pid_controller = PIDController(*self.pid_params, torch.zeros(1, self.n,1), self.dt)
                # Load model
        if load_model:
            controller: RecoveryNetwork = torch.load(load_model_path, map_location=device) 
        else:
            controller: RecoveryNetwork = torch.load(path_results+'best-model.pt', map_location=device) 

        controller.set_batchsize(1)
        controller.InitSequence(self.m1x_0.unsqueeze(0), T)
        controller.init_hidden_KNet()
        # Pre allocate an array for current state
        self.x = torch.zeros(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.zeros(size=[self.n, T])
        # Pre allocate an array for current control signal
        self.u = torch.zeros(size=[self.p, T])
        # Pre allocate an array for current setpoint
        self.s = torch.zeros(size=[self.p, T]) 
        # Set x0 to be x previous
        self.x_prev = self.m1x_0.unsqueeze(0)
        xt = self.x_prev
        
        ut = torch.zeros(1, self.p, 1)

        # Randomized mission: setpoints change over time
        setpoint = torch.ones(1, self.p, 1)*self.init_setpoint
        setpoint = setpoint + torch.rand(1,self.p, 1)*(self.init_setpoint/4) - torch.ones(1,self.p, 1)*(self.init_setpoint/8)
        for t in range(T):
            if t % (T // 4) == 0:  # Change setpoints at intervals
                setpoint = setpoint + torch.rand(1,self.p, 1)*self.setpoint_change - torch.ones(1,self.p, 1)*(self.setpoint_change/2)  # New target in [-1,1]
                # self.pid_controller.set_setpoint(setpoint)
            
            # State evolution
            if torch.equal(Q_gen, torch.zeros(self.m, self.m)):
                xt = self.f(self.x_prev, ut)
            elif self.m == 1: # 1 dim noise
                xt = self.f(self.x_prev, ut)
                eq = torch.normal(mean=0, std=Q_gen)
                # Additive Process Noise
                xt = torch.add(xt,eq)
            else:
                xt = self.f(self.x_prev, ut)
                mean = torch.zeros([self.m])              
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                eq = distrib.rsample()
                eq = torch.reshape(eq[:], xt.size())
                # Additive Process Noise
                xt = torch.add(xt,eq)
            
            # Observation with noise
            yt = self.h(xt)
            if torch.equal(R_gen, torch.zeros(self.n, self.n)):
                yt = self.h(xt)
            if self.n == 1: # 1 dim noise
                er = torch.normal(mean=0, std=R_gen)
                # Additive Observation Noise
                yt = torch.add(yt,er)
            else:
                mean = torch.zeros([self.n])            
                distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
                er = distrib.rsample()
                er = torch.reshape(er[:], yt.size())       
                # Additive Observation Noise
                yt = torch.add(yt,er)
            
            yt = (yt-scaler_min) / (scaler_max-scaler_min)
            # Compute control input using PID
            ut, estimated_state = controller(yt, setpoint)
            
            # Save data
            self.x[:, t] = xt.squeeze()
            self.y[:, t] = yt.squeeze()
            self.u[:, t] = ut.squeeze()
            self.s[:, t] = setpoint.squeeze()
            self.x_prev = xt
        
        return self.x, self.y, self.u, self.s

if __name__ == "__main__":
    import config
    from Simulations.Hillclimbing_Car.parameters import m1x_0, m2x_0, m, n, p, pid_params, dt, \
    f, h, Q_structure, R_structure, init_setpoint, setpoint_change
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
    run_control_system = False

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
    sys_model = ControlSystemModel(f, Q, h, R, args.T, args.T_test, m, n, p, pid_params, dt, init_setpoint, setpoint_change)# parameters for GT
    sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0

    states, measurements, controls, setpoints = sys_model.GenerateSequenceRecovery(
        Q_gen=Q,
        R_gen=R,
        T=args.T,
        scaler_min=torch.tensor(7.5792),
        scaler_max=torch.tensor(26.6838),
        path_results=path_results,
        device=torch.device('cpu')
    )