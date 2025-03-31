import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from .Extended_sysmdl import SystemModel
from Controllers.PID import PIDController
from typing import List

class ControlSystemModel(SystemModel):
    def __init__(self, f, Q, h, R, T, T_test, m, n, p, pid_params, dt, prior_Q=None, prior_Sigma=None, prior_S=None):
        super().__init__(f, Q, h, R, T, T_test, m, n, prior_Q, prior_Sigma, prior_S)
        self.p = p      # Dimension of control signal
        self.pid_params = pid_params
        self.dt = dt

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
        setpoint = torch.zeros(self.p, 1)
        for t in range(T):
            if t % (T // 4) == 0:  # Change setpoints at intervals
                setpoint = torch.rand(self.p, 1) * 2 - 1  # New target in [-1,1]
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

            # Set x0 to be x previous
            self.x_prev = self.m1x_0_batch
            xt = self.x_prev
            ut = torch.zeros(size, self.p, 1)
            # Generate in a batched manner
            for t in range(0, T):
                ########################
                #### New Setpoint ######
                if t % (T // 4) == 0:  # Change setpoints at intervals
                    setpoint = torch.rand(size, self.p, 1) * 2 - 1  # New target in [-1,1]
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

                ################################
                ### Save Current to Previous ###
                ################################
                self.x_prev = xt
    

if __name__ == "__main__":
    sys_model = ControlSystemModel()
