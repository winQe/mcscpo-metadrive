import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import numpy as np
import torch
from torch.optim import Adam
from scipy.sparse.linalg import LinearOperator, cg as scipy_cg

from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
import gymnasium as gym
from metadrive.envs.gym_wrapper import createGymWrapper # import the wrapper
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod

import time
import copy
import mcscpo_core as core
import qp_solver as solver

from utils.logx import EpochLogger, setup_logger_kwargs, colorize
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, mpi_sum

import os.path as osp
import warnings

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-8

class SCPOBuffer:
    """
    A buffer for storing trajectories experienced by a SCPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, num_constraints, gamma=0.99, lam=0.95, cgamma=1., clam=0.95):
        self.obs_buf      = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf      = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf      = np.zeros(size, dtype=np.float32)
        self.rew_buf      = np.zeros(size, dtype=np.float32)
        self.ret_buf      = np.zeros(size, dtype=np.float32)
        self.val_buf      = np.zeros(size, dtype=np.float32)
        self.cost_buf     = np.zeros((size,num_constraints), dtype=np.float32) # D buffer for multi-constraints
        self.cost_ret_buf = np.zeros((size,num_constraints), dtype=np.float32) # D return buffer for multi constraints
        self.cost_val_buf = np.zeros((size,num_constraints), dtype=np.float32) # Vd buffer for multi constraints
        self.adc_buf      = np.zeros((size,num_constraints), dtype=np.float32) # Advantage of cost D multi constraints
        self.logp_buf     = np.zeros(size, dtype=np.float32)
        self.mu_buf       = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.logstd_buf   = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.gamma, self.lam = gamma, lam # lam -> for GAE
        self.cgamma, self.clam = cgamma, clam # there is no discount for the cost for MMDP 
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, cost, cost_val, mu, logstd):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr]      = obs   # augmented state space
        self.act_buf[self.ptr]      = act   # actions (vector of probabilities)
        self.rew_buf[self.ptr]      = rew   # reward
        self.val_buf[self.ptr]      = val   # value function return at current (s,t)
        self.logp_buf[self.ptr]     = logp
        self.cost_buf[self.ptr]     = cost  # actual cost received at timestep
        self.cost_val_buf[self.ptr] = cost_val # D value (we learn D not Jc here I guess) 
        self.mu_buf[self.ptr]       = mu
        self.logstd_buf[self.ptr]   = logstd
        self.ptr += 1

    def finish_path(self, last_val, last_cost_val):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        costs = np.vstack((self.cost_buf[path_slice], last_cost_val))
        cost_vals = np.vstack((self.cost_val_buf[path_slice], last_cost_val))
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam) # A
        
        # cost advantage calculation
        cost_deltas = costs[:-1] + self.cgamma * cost_vals[1:] - cost_vals[:-1]
        self.adc_buf[path_slice] = core.discount_cumsum(cost_deltas, self.cgamma * self.clam) # AD
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1] # Actual return
        
        # costs-to-go, targets for the cost value function
        self.cost_ret_buf[path_slice] = core.discount_cumsum(costs, self.cgamma)[:-1] # Actual D
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick, std = 1, mean = 0
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        # center cost advantage, but don't scale

        adc_mean = np.zeros(self.adc_buf.shape[1])
        adc_std = np.zeros(self.adc_buf.shape[1])

        for i in range(self.adc_buf.shape[1]):
            column_data = self.adc_buf[:, i]
            adc_mean[i], adc_std[i] = mpi_statistics_scalar(column_data)
        
        self.adc_buf = self.adc_buf - adc_mean[np.newaxis, :]

        data = dict(obs=torch.FloatTensor(self.obs_buf).to(device), 
                    act=torch.FloatTensor(self.act_buf).to(device), 
                    ret=torch.FloatTensor(self.ret_buf).to(device),
                    adv=torch.FloatTensor(self.adv_buf).to(device),
                    cost_ret=torch.FloatTensor(self.cost_ret_buf).to(device),
                    adc=torch.FloatTensor(self.adc_buf).to(device),
                    logp=torch.FloatTensor(self.logp_buf).to(device),
                    mu=torch.FloatTensor(self.mu_buf).to(device),
                    logstd=torch.FloatTensor(self.logstd_buf).to(device))
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


def get_net_param_np_vec(net):
    """
        Get the parameters of the network as numpy vector
    """
    return torch.cat([val.flatten() for val in net.parameters()], axis=0).detach().cpu().numpy()

def assign_net_param_from_flat(param_vec, net):
    param_sizes = [np.prod(list(val.shape)) for val in net.parameters()]
    ptr = 0
    for s, param in zip(param_sizes, net.parameters()):
        param.data.copy_(torch.from_numpy(param_vec[ptr:ptr+s]).reshape(param.shape))
        ptr += s

def cg(Ax, b, Mx=None, x=None, cg_iters=250, tol=1e-2):
    """
    A custom conjugate gradient solver with preconditioning.

    Parameters:
    - Ax: Function that computes the matrix-vector product Ax.
    - b: Right-hand side vector of the equation Ax = b.
    - Mx: Preconditioner function. If None, a Jacobi preconditioner is created.
    - x: Initial guess for the solution (default is a zero vector).
    - cg_iters: Maximum number of iterations.
    - tol: Tolerance for convergence.

    Returns:
    - x: The computed solution.
    """
    if x is None:
        x = np.zeros_like(b)

    n = len(b)
    A_linop = LinearOperator((n, n), matvec=Ax)

    # Define the preconditioner as a linear operator
    if Mx is None:
        # Estimate the diagonal of the matrix for Jacobi preconditioner
        diag = np.zeros(n)
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1
            A_e_i = Ax(e_i)
            diag[i] = A_e_i[i]

        # Add a small constant to avoid division by zero
        epsilon = 1e-10
        diag_safe = diag + epsilon

        # Invert the diagonal for the preconditioner
        M_inv = 1.0 / diag_safe
        Mx = lambda x: M_inv * x
        M_linop = LinearOperator((n, n), matvec=Mx)
    else:
        M_linop = LinearOperator((n, n), matvec=Mx)

    # Use SciPy's conjugate gradient solver with preconditioner
    x, info = scipy_cg(A=A_linop, b=b, x0=x, tol=tol, maxiter=cg_iters, M=M_linop)

    if info > 0:
        print(f"Convergence not achieved after {info} iterations.")
    elif info < 0:
        print("Illegal input or breakdown.")
    else:
        print(f"Convergence achieved.")

    return x

def auto_grad(objective, net, to_numpy=True):
    """
    Get the gradient of the objective with respect to the parameters of the network
    """
    grad = torch.autograd.grad(objective, net.parameters(), create_graph=True)
    if to_numpy:
        return torch.cat([val.flatten() for val in grad], axis=0).detach().cpu().numpy()
    else:
        return torch.cat([val.flatten() for val in grad], axis=0)


def auto_hession_x(objective, net, x):
    """
    Returns 
    """
    jacob = auto_grad(objective, net, to_numpy=False)
    
    return auto_grad(torch.dot(jacob, x), net, to_numpy=True)

def calculate_cost(value):
    if value < 4:
        return 0.0
    elif value <= 20:
        # Linear increase from 0 at value = 4 to a certain point at value = 20
        return (value - 4) / (20 - 4) * 0.2  # Adjusted to ensure smooth transition
    else:
        # For values above 20, use a function that increases continuously. 
        # We can use a logarithmic scale adjusted to ensure it fits well within our desired output range.
        # This example uses a logarithm to ensure uniqueness but it's a simple placeholder for demonstration.
        # Adjust the base and scaling to fit within the 0.2 to 1.0 range more appropriately.
        return 0.5 * (0.2 + (np.log(value - 19) / np.log(40 - 19)) * (1.0 - 0.2))

def scpo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, vcf_lr=1e-3, train_v_iters=80, train_vc_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, target_cost = 0.0, logger_kwargs=dict(), save_freq=10, backtrack_coeff=0.8, 
        backtrack_iters=100, model_save=True, cost_reduction=0, num_constraints=3):
    """
    State-wise Constrained Policy Optimization, 
 
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.
        
        vcf_lr (float): Learning rate for cost value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.
            
        train_vc_iters (int): Number of gradient descent steps to take on 
            cost value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)
            
        target_cost (float): Cost limit that the agent should satisfy

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
            
        backtrack_coeff (float): Scaling factor for line search.
        
        backtrack_iters (int): Number of line search steps.
        
        model_save (bool): If saving model.
        
        cost_reduction (float): Cost reduction imit when current policy is infeasible.

        num_constraints (int): Number of constraints to be enforced

    """
    cost_reduction = np.full(num_constraints, cost_reduction)
    model_save=True
    assert len(target_cost) == num_constraints

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn() 
    # Augmented state space here
    obs_dim = (env.observation_space.shape[0]+num_constraints,) # this is especially designed for SCPO, since we require an additional M in the observation space 
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, num_constraints, **ac_kwargs).to(device)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = SCPOBuffer(obs_dim, act_dim, local_steps_per_epoch, num_constraints, gamma, lam)
    
    def compute_kl_pi(data, cur_pi):
        """
        Return the sample average KL divergence between old and new policies
        """
        obs, act, adv, logp_old, mu_old, logstd_old = data['obs'], data['act'], data['adv'], data['logp'], data['mu'], data['logstd']
        # Average KL Divergence  
        # pi, logp = cur_pi(obs, act)
        # average_kl = (logp_old - logp).mean()
        average_kl = cur_pi._d_kl(
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(mu_old, dtype=torch.float32),
            torch.as_tensor(logstd_old, dtype=torch.float32), device=device)
        
        return average_kl
    
    def compute_cost_pi(data, cur_pi):
        """
        Return the suggorate cost for current policy
        """
        obs, act, adc, logp_old = data['obs'], data['act'], data['adc'], data['logp']
        
        # Surrogate cost function, D cost not C
        pi, logp = cur_pi(obs, act)
        ratio = torch.exp(logp - logp_old).unsqueeze(-1)# Make ratio broadcast-compatible with adc
        surr_cost = (ratio * adc).sum(dim=0) #different from cpo one, should still equate to mean
        epochs = len(logger.epoch_dict['EpCost'])
        surr_cost /= epochs # the average 
        
        return surr_cost
        
        
    def compute_loss_pi(data, cur_pi):
        """
        The reward objective for SCPO (SCPO policy loss)
        """
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        # Policy loss 
        pi, logp = cur_pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        loss_pi = -(ratio * adv).mean()
        
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)
        
        return loss_pi, pi_info
        
    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()
    
    # Set up function for computing cost loss 
    def compute_loss_vc(data, index, vc_net):
        obs, cost_ret = data['obs'], data['cost_ret'][:,index]

        # Split the data into positive and zero cost returns.
        # This is to address potential imbalance in the dataset.
        cost_ret_positive = cost_ret[cost_ret > 0]
        obs_positive = obs[cost_ret > 0]
        cost_ret_zero = cost_ret[cost_ret == 0]
        obs_zero = obs[cost_ret == 0]
        
        if len(cost_ret_zero) > 0:
            # Calculate the fraction of positive returns to zero returns
            frac = len(cost_ret_positive) / len(cost_ret_zero) 
            
            # If there are fewer positive returns than zero returns
            if frac < 1. :# Fraction of elements to keep
                # Randomly down-sample the zero returns to match the number of positive returns.
                indices = np.random.choice(len(cost_ret_zero), size=int(len(cost_ret_zero)*frac), replace=False)
                cost_ret_zero_downsample = cost_ret_zero[indices]
                obs_zero_downsample = obs_zero[indices]
                
                # Combine the positive and down-sampled zero returns
                obs_downsample = torch.cat((obs_positive, obs_zero_downsample), dim=0)
                cost_ret_downsample = torch.cat((cost_ret_positive, cost_ret_zero_downsample), dim=0)
            else:
                # If there's no need to down-sample, use the entire dataset
                obs_downsample = obs
                cost_ret_downsample = cost_ret
        else:
            # If there are no zero returns in the dataset, use the entire dataset
            obs_downsample = obs
            cost_ret_downsample = cost_ret
        # Calculate and return the mean squared error loss between the cost network and the actual cost return
        return ((vc_net(obs_downsample) - cost_ret_downsample)**2).mean()

    get_costs = lambda info, constraints: np.array([info[key] for key in constraints])
    get_d = lambda info, constraints: np.array([info[key] for key in constraints if key != 'cost'])

    
    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    vcf_optimizers = [Adam(vc.parameters(), lr=vcf_lr) for vc in ac.vcs]

    # Set up model saving
    if model_save:
        logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        # log the loss objective and cost function and value function for old policy
        pi_l_old, pi_info_old = compute_loss_pi(data, ac.pi)
        pi_l_old = pi_l_old.item()
        surr_cost_old = compute_cost_pi(data, ac.pi).detach().cpu().numpy()
        # surr_cost_old = surr_cost_old.item()
        v_l_old = compute_loss_v(data).item()

        # SCPO policy update core impelmentation 
        loss_pi, pi_info = compute_loss_pi(data, ac.pi)
        surr_cost = compute_cost_pi(data, ac.pi)
        
        # get Hessian for KL divergence
        kl_div = compute_kl_pi(data, ac.pi)
        
        # Compute dot product of Hessian matrix with x
        Hx = lambda x: auto_hession_x(kl_div, ac.pi, torch.FloatTensor(x).to(device))
        
        # linearize the loss objective and cost function
        g = auto_grad(loss_pi, ac.pi) # get the loss flatten gradient evaluted at pi old 
        B = np.array([auto_grad(cost_elem, ac.pi) for cost_elem in surr_cost]) # get the cost increase flatten gradient evaluted at pi old
        # get the Episode cost
        EpMaxCost = np.array(logger.get_value('EpMaxCost')) #EpMaxCost = M, different compared to EpCost
        # cost constraint linearization
        '''
        original fixed target cost, in the context of mean adv of epochs
        '''
        # c = EpMaxCost - target_cost 
        # rescale  = EpLen
        # c /= (rescale + EPS)
        
        '''
        fixed target cost, in the context of sum adv of epoch
        '''
        # if negative means M (maximum statewise cost) cost limit, thus infeasible
        c = np.squeeze(EpMaxCost - np.array(target_cost))
        c = np.atleast_1d(c)
        # core calculation for SCPO
        # # Conjugate Gradient to calculate H^-1
        Hinv_g   = cg(Hx, g)             # Hinv_g = H^-1 * g        
        approx_g = Hx(Hinv_g)           # g
        # Analytical solution from the CPO paper (Appendix 10.2)
        # q = g.T * H^-1 * g
        q        = Hinv_g.T @ approx_g

        Hinv_B = np.array([cg(Hx, b) for b in B]) # std is a bit too much here
        approx_B = np.array([Hx(Hinv_b) for Hinv_b in Hinv_B])

        # Assuming Hx, B, Hinv_B, and approx_B are already defined
        r = Hinv_B @ approx_g          # b^T H^{-1} g
        S =  approx_B @ Hinv_B.T      # b^T H^{-1} b

        # Start timing
        print("g approximation error ", np.linalg.norm(approx_g - g))
        for i in range(len(B)):
            print("b{} approximation error ".format(i), np.linalg.norm(approx_B[i] - B[i]))

        solver_start = time.time()

        # Use QP library to solve the QP
        # Setup QP Solver
        qp_solver = solver.QuadraticOptimizer(num_constraints)
        qp_solver.solve(c,q,r,S,target_kl)
        lam,nu,status = qp_solver.get_solution()

        if status == "Infeasible":
            # Recovery scheme
            # Find the direction that purely decreases total cost
            # Sum all the cost gradient, perform similar recovery to the SCPO paper
            # Will decrease total cost but might increase one of the cost
            b_sum = np.sum(B, axis=0).ravel()
            Hinv_b_sum = cg(Hx,b_sum)
            approx_b_sum = Hx(Hinv_b_sum)
            s_sum = Hinv_b_sum.T @ approx_b_sum

            nu = np.sqrt(2 * target_kl / (s_sum+EPS))
            x_direction = nu * Hinv_b_sum
            logger.store(Infeasible=1)

        else:
            x_direction = (1./(lam+EPS)) * (Hinv_g + nu @ Hinv_B)
            logger.store(Infeasible=0)
               
        # Stop timing
        solver_end = time.time()

        np.set_printoptions(precision=4, suppress=True)
        print("lambda and nu value from solver = [{},{}]".format(lam,nu))
        print(f"Time taken: {solver_end - solver_start} seconds")

        # copy an actor to conduct line search 
        actor_tmp = copy.deepcopy(ac.pi)
        def set_and_eval(step):
            new_param = get_net_param_np_vec(ac.pi) - step * x_direction
            assign_net_param_from_flat(new_param, actor_tmp)
            kl = compute_kl_pi(data, actor_tmp)
            pi_l, _ = compute_loss_pi(data, actor_tmp)
            surr_cost = compute_cost_pi(data, actor_tmp)
            
            return kl, pi_l, surr_cost
        
        def is_cost_within_threshold(new_cost, old_cost, threshold):
         return np.all(new_cost - old_cost <= threshold)

        def is_sum_cost_within_threshold(new_cost, old_cost, threshold):
            return np.sum(new_cost - old_cost) <= threshold
        
        # update the policy such that the KL diveragence constraints are satisfied and loss is decreasing
        # backtracking line search to enforce constraint satisfaction
        for j in range(backtrack_iters):
            try:
                kl, pi_l_new, surr_cost_new = set_and_eval(backtrack_coeff**j)
            except:
                import ipdb; ipdb.set_trace()
            
            # Define conditions for clarity
            is_feasible = status != "Infeasible"
            is_infeasible = status == "Infeasible"
            kl_within_target = kl.item() <= target_kl
            policy_loss_improved = pi_l_new.item() <= pi_l_old # if current policy is feasible (optim>1), must preserve pi loss
            cost_within_threshold = is_cost_within_threshold(surr_cost_new.detach().cpu().numpy(), surr_cost_old, np.maximum(-c, -cost_reduction))
            sum_cost_within_threshold = is_sum_cost_within_threshold(surr_cost_new.detach().cpu().numpy(), surr_cost_old, max(-np.sum(c), -np.sum(cost_reduction)))

            # Refactored conditional statement
            if ((is_feasible and kl_within_target and policy_loss_improved and cost_within_threshold) or
                (is_infeasible and sum_cost_within_threshold)):
                print(colorize(f'Accepting new params at step %d of line search.'%j, 'green', bold=False))

                # update the policy parameter 
                new_param = get_net_param_np_vec(ac.pi) - backtrack_coeff**j * x_direction
                assign_net_param_from_flat(new_param, ac.pi)
                
                loss_pi, pi_info = compute_loss_pi(data, ac.pi) # re-evaluate the pi_info for the new policy
                surr_cost = compute_cost_pi(data, ac.pi) # re-evaluate the surr_cost for the new policy
                break
            if j==backtrack_iters-1:
                print(colorize(f'Line search failed! Keeping old params.', 'yellow', bold=False))

        # Value function learning
        for _ in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()
            
        # Cost value function learning
        for _ in range(train_vc_iters):
            index = 0
            for optimizer, vc_net in zip(vcf_optimizers, ac.vcs):
                optimizer.zero_grad()
                loss_vc = compute_loss_vc(data,index,vc_net)  # Assuming compute_loss_vc can take a specific vc as an argument
                loss_vc.backward()
                mpi_avg_grads(vc_net)  # average grads across MPI processes for the specific vc
                optimizer.step()
                index += 1

        # Log changes from update        
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=pi_l_old, LossV=v_l_old, LossCost=surr_cost_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old),
                     DeltaLossCost=(surr_cost.detach().cpu().numpy() - surr_cost_old))

    # Prepare for interaction with environment
    start_time = time.time()
    
    while True:
        try:
            o, ep_ret, ep_len = env.reset(), 0, 0
            break
        except:
            print('reset environment is wrong, try next reset')
    
    # Initialize the environment and cost all = 0
    ep_cost_ret = np.zeros(num_constraints + 1, dtype=np.float32)
    ep_cost = np.zeros(num_constraints + 1, dtype=np.float32)
    cum_cost = 0
    M = np.zeros(num_constraints, dtype=np.float32) # initialize the maximum cost a 0 per constraints
    o_aug = np.append(o, M) # augmented observation = observation + M 
    first_step = True
    constraints_list = []
    first_epoch = True

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        trajectory_index = 0
        for t in range(local_steps_per_epoch):
            # Forward, get action and value estimates (cost and reward) for the current observation
            a, v, vcs, logp, mu, logstd = ac.step(torch.as_tensor(o_aug, dtype=torch.float32))
            
            try: 
                next_o, r, d, i = env.step(a)
                info = dict()

                if num_constraints == 3:
                  #Positional cost
                    info['cost_out_of_road'] = i.get('cost_out_of_road', 0)
                    info['crash_vehicle_cost'] = i.get('crash_vehicle_cost', 0)
                    info['crash_object_cost'] = i.get('crash_object_cost', 0)
                    info['cost'] = info['cost_out_of_road'] + info['crash_vehicle_cost'] + info['crash_object_cost']
                elif num_constraints == 20:
                    info['cost_env'] = i.get('cost_out_of_road', 0) + i.get('crash_vehicle_cost', 0) + i.get('crash_object_cost', 0)
                    info['cost_acceleration'] = calculate_cost(i['accel_ms'])
                    info['cost_jerk'] = calculate_cost(i['jerk'])
                    info['cost'] = info['cost_env'] + info['cost_acceleration'] + info['cost_jerk']
                assert 'cost' in info.keys()
            except: 
                # simulation exception discovered, discard this episode 
                next_o, r, d = o, 0, True # observation will not change, no reward when episode done
                for cost in info.keys():
                    info[cost] = 0 
                # no cost when episode done
            if first_epoch:
                constraints_list = [key for key in info.keys()]
                first_epoch = False
            if first_step:
                # the first step of each episode
                cost_increase =  get_d(info, constraints_list)
                # cost_increase = info['cost'] # define the new observation and cost for Maximum Markov Decision Process
                M_next = cost_increase
                first_step = False
            else:
                # the second and forward step of each episode
                # cost increase = D, to be constrained to ensure state-wise safety, constraining maximum violation in state transition -> enforcing statewise safety
                try:
                    costs_D = np.array(get_d(info, constraints_list))
                except:
                    import ipdb; ipdb.set_trace()
                    # Handle exceptions here
                cost_increase = np.maximum(costs_D - M, 0)
                M_next = M + cost_increase
             
            # Track cumulative cost over training
            cum_cost += info['cost'] # not equal to M
            ep_ret += r
            ep_cost_ret += get_costs(info, constraints_list) * (gamma ** t)
            ep_cost += get_costs(info, constraints_list)
            ep_len += 1

            # save and log, buffer is different, store 
            buf.store(o_aug, a, r, v, logp, cost_increase, vcs, mu, logstd)
            logger.store(VVals=v)
            
            # Update obs (critical!)
            # o = next_o
            M = M_next
            o_aug = np.append(next_o, M_next)

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _, _, _, _ = ac.step(torch.as_tensor(o_aug, dtype=torch.float32))
                    vc = np.zeros(num_constraints,dtype=np.float32)
                else:
                    v = 0
                    vc = np.zeros(num_constraints,dtype=np.float32)

                buf.finish_path(v, vc)
                if terminal:
                    # only save EpRet / EpLen / EpCostRet if trajectory finished
                    # EpMaxCost = Max MDP M, while EpCost is just CMDP, it's Maximum state-wise cost, cannot be canceled out with negative cost if that even exist
                    # 1 epoch containts lots of path
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCostRet=ep_cost_ret, EpCost=ep_cost, EpMaxCost=M)
                    trajectory_index += 1
                while True:
                    try:
                        o, ep_ret, ep_len = env.reset(), 0, 0
                        break
                    except:
                        print('reset environment is wrong, try next reset')
                ep_cost_ret = np.zeros(num_constraints + 1, dtype=np.float32)
                ep_cost = np.zeros(num_constraints + 1, dtype=np.float32)
                M = np.zeros(num_constraints, dtype=np.float32) # initialize the maximum cost a 0 per constraints
                o_aug = np.append(o, M) # augmented observation = observation + M 
                first_step = True

        # Save model
        if ((epoch % save_freq == 0) or (epoch == epochs-1)) and model_save:
            logger.save_state({'env': env}, None)

        # Perform SCPO update!
        update()
        
        #=====================================================================#
        #  Cumulative cost calculations                                       #
        #=====================================================================#
        cumulative_cost = mpi_sum(cum_cost)
        cost_rate = cumulative_cost / ((epoch+1)*steps_per_epoch)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('EpCostRet', average_only=True)
        logger.log_tabular('EpCost', average_only=True)
        logger.log_tabular('EpMaxCost', average_only=True)
        logger.log_tabular('CumulativeCost', cumulative_cost)
        logger.log_tabular('CostRate', cost_rate)
        logger.log_tabular('VVals', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('LossCost', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('DeltaLossCost', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.log_tabular('Infeasible', average_only=True)
        logger.log_tabular('NumberOfTrajectories',trajectory_index)
        logger.dump_tabular()
        
        
def create_env(map_type):
    map_config = {
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
        BaseMap.GENERATE_CONFIG: "X",  # 3 block
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 2,
    }
    map_config["config"]= map_type
    lidar=dict(
        num_lasers=50, distance=50, num_others=0, gaussian_noise=0.0, dropout_prob=0.0, add_others_navi=False
    )
    vehicle_config = dict(lidar=lidar)

    # env = SafeMetaDriveEnv(dict(map_config = map_config))
    env = createGymWrapper(SafeMetaDriveEnv)(
        config={
            "use_render": False,
            "map_config": map_config,
            "vehicle_config": vehicle_config,
            "num_scenarios": 1,
            "accident_prob": 0.8,
            "start_seed": 100,
            "crash_vehicle_done": False,
            "crash_object_done": False,
            "out_of_route_done": False,
            "cost_to_reward": False,
            "crash_vehicle_penalty": 0.0,
            "crash_object_penalty": 0.0,
            "traffic_density": 0.55,
        }
    )  # wrap the environment
    return env

def parse_float_list(s):
    return [float(item) for item in s.split(',')]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()    
    parser.add_argument('--map_type', type=str, default='SXCOY')
    parser.add_argument('--target_cost', type=parse_float_list, default=[0.00,0.00,0.00]) # the array of cost limit for the environment
    parser.add_argument('--target_kl', type=float, default=0.02) # the kl divergence limit for SCPO
    parser.add_argument('--cost_reduction', type=float, default=0.) # the cost_reduction limit when current policy is infeasible
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=30000)                                       
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--exp_name', type=str, default='solverbased_scpo')
    parser.add_argument('--model_save', action='store_true')
    parser.add_argument('--num_constraints', type=int, default=1) # Number of constraints

    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    
    exp_name = args.exp_name + '_' + args.map_type \
                + '_' + 'constraints' + str(args.num_constraints) \
                + '_' + 'epoch' + str(args.epochs)
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    # whether to save model
    model_save = True if args.model_save else False

    scpo(lambda : create_env(args.map_type), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, target_cost=args.target_cost, 
        model_save=model_save, target_kl=args.target_kl, cost_reduction=args.cost_reduction,
        num_constraints=args.num_constraints, max_ep_len=1000)