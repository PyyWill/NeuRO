import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer
import torch

class POPersuitOptimization:
    def __init__(self, batch_size=None, T=4, edge=None, gamma=0.95, N=None):
        self.batch_size = batch_size
        self.T = T
        self.edge = edge
        self.area = edge * edge
        self.gamma = gamma
        self.N = N
        self.init_pos = self.area // 2
        self.beta, self.alpha = {}, {}
        self.M = [cp.Parameter((self.area, self.area), name=f'M_{n}') for n in range(self.N)]

        for n in range(self.N):
            self.beta[n] = cp.Variable((self.T, self.area+1))
            self.alpha[n] = cp.Variable((self.T, self.area))

        self.p = cp.Variable((self.T, 2))  # [0, 0] - [edge, edge]
        self.z = cp.Variable((self.T, self.area))

    def optimization_model(self):
        constraints = []
        params = []
        for n in range(self.N):
            params += [self.beta[n], self.alpha[n]]

        # [0, 1]-constraint
        for param in params:
            constraints.append(param >= 0)
            constraints.append(param <= 1)

        for t in range(self.T):
            constraints += [
                self.p[t][0] <= self.edge, self.p[t][1] <= self.edge,
                self.p[t][0] >= 0, self.p[t][1] >= 0
            ]
        constraints.append(self.p[0][0] == 1.5)
        constraints.append(self.p[0][1] == 1.5)

        for v in list(range(self.init_pos)) + list(range(self.init_pos + 1, self.area)):
            for n in range(self.N):
                constraints.append(self.beta[n][0, v+1] == 1/(self.area-1))
        for t in range(self.T-1):
            constraints.append(cp.norm(self.p[t+1] - self.p[t], 2) <= 1)

        for t in range(self.T-1):
            for v in range(self.area):
                for n in range(self.N):
                    constraints.append(self.alpha[n][t+1, v] == cp.sum([
                        self.M[n][u][v] * self.beta[n][t, u+1] for u in range(self.area)
                    ]))

        def index_to_xy(index):
            return (index % self.edge + 0.5), (index // self.edge + 0.5)

        for t in range(self.T):
            for v in range(self.area):
                for n in range(self.N):
                    constraints.append(
                        (cp.abs(index_to_xy(v)[0] - self.p[t][0]) + cp.abs(index_to_xy(v)[1] - self.p[t][1])) / (self.area-1) <= self.z[t, v]
                    )
                    constraints.append(self.beta[n][t, v+1] <= self.z[t, v])
                    constraints.append(self.beta[n][t, v+1] <= self.alpha[n][t, v])
                    constraints.append(self.beta[n][t, v+1] >= self.alpha[n][t, v] - (1 - self.z[t, v]))

        for t in range(self.T):
            for n in range(self.N):
                constraints.append(self.beta[n][t, 0] == (1 - cp.sum(self.beta[n][t, 1:])))
        objective_func = cp.sum([
            self.gamma**t * (self.beta[n][t, 0]) for t in range(self.T) for n in range(self.N)
        ])
        objective = cp.Maximize(objective_func)
        self.problem = cp.Problem(objective, constraints)

    def torch_layer(self):
        self.cvxpylayer = CvxpyLayer(self.problem, parameters=self.M, variables=[self.p, self.beta[0], self.beta[1]])

    def optimization_loss(self, beta_vals):
        bs = self.batch_size
        loss = []
        for b in range(bs):
            loss.append(torch.sum(torch.stack([self.gamma**t * beta_vals[n][b, t, 0] for t in range(self.T) for n in range(self.N)])))
        loss = torch.stack(loss)
        loss = torch.mean(loss)
        
        return loss
    
    def run_optimization(self, M_val=None):
        def map_delta_action(delta_action):
            result = torch.zeros(delta_action.shape[0], dtype=torch.long)
            
            mask1 = delta_action[:, 1] < 0
            result[mask1] = 1
            mask4 = delta_action[:, 1] > 0
            random_assignments = torch.randint(2, 4, (mask4.sum(),))  
            result[mask4] = random_assignments
            mask0 = torch.abs(delta_action[:, 1]) < 1e-3
            mask2 = (delta_action[:, 0] < 0) & mask0  
            mask3 = (delta_action[:, 0] > 0) & mask0  
            result[mask2] = 2
            result[mask3] = 3

            return result
        
        beta_vals = {}
        p_vals, beta_vals[0], beta_vals[1] = self.cvxpylayer(*M_val)
        delta_action = p_vals[:, 1] - p_vals[:, 0]
        mapped_values = map_delta_action(delta_action).unsqueeze(-1)
        
        loss = self.optimization_loss(beta_vals)

        return mapped_values, loss


