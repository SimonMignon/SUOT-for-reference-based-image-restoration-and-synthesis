import torch
from pykeops.torch import LazyTensor

# Softmin function
def softmin(ε, x ,y , f): # -εlog(sum(exp()))

    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])
    f_j = LazyTensor(f[None,:,None])
    
    D_ij = ((x_i - y_j) ** 2).sum(-1)
    
    smin = (f_j - D_ij * torch.Tensor([1 / ε]).type_as(x)).logsumexp(1, enable_chunks=False).view(-1)
    return - ε * smin

# ROT Sinkhorn ########################################

def ROT(x,y,ε,fg_init=None,nb_it=10,dev="cuda"):
    N= x.shape[0]
    M= y.shape[0]
    
    # no grad required for the optimisation part
    torch.autograd.set_grad_enabled(False)
    
    # Distributions
    α = 1 /N * torch.ones(x.shape[0], dtype=torch.float32).to(device=dev)
    β = 1 /M * torch.ones(y.shape[0], dtype=torch.float32).to(device=dev)
    log_α=torch.log(α)
    log_β=torch.log(β)
    
    # ε schedule
    ε_s=[ε]*nb_it
    last_iteration=len(ε_s)-1

    # initialise dual variables
    if fg_init==None:
        f = torch.zeros_like(α)
        g = torch.zeros_like(β)
    if fg_init!=None:
        f = fg_init[0]
        g = fg_init[1]
    
    # Optimisation of dual variables
    for i, ε in enumerate(ε_s):  
        f,g= 0.5*(softmin(ε, x, y, log_β + g/ε)+f), 0.5*(softmin(ε, y, x, log_α + f/ε)+g)

        if i==last_iteration: # prepare gradient for autodiff
            torch.autograd.set_grad_enabled(True)
            g = softmin(ε, y,x, (log_α + f/ε)).detach()
            f = softmin(ε, x,y, (log_β + g/ε).detach())
                        
    # ROT cost
    OT_ε= torch.sum(α*f)+torch.sum(β*g)

    return OT_ε,[f.detach(),g.detach()]

# RSUOT Sinkhorn ########################################

def RSUOT(x,y,ε,ρ,f_init=None,nb_it=10,dev="cuda"):
    N= x.shape[0]
    M= y.shape[0]
    
    # no grad required for the optimisation part
    torch.autograd.set_grad_enabled(False)
    
    # Distributions
    α = 1 /N * torch.ones(x.shape[0], dtype=torch.float32).to(device=dev)
    β = 1 /M * torch.ones(y.shape[0], dtype=torch.float32).to(device=dev)
    log_α=torch.log(α)
    log_β=torch.log(β)
    
    # ε schedule
    ε_s=[ε]*nb_it
    last_iteration=len(ε_s)-1
    
    # initialise dual variables
    if f_init==None:
        f = torch.zeros_like(α)
    if f_init!=None:
        f = f_init
    
    # Optimisation of dual variables
    λ = 1 / (1 + ε / ρ)
    for i, ε in enumerate(ε_s): 
        g = λ*softmin(ε, y,x, (log_α + f/ε))
        f = softmin(ε, x,y, (log_β + g/ε))

        if i==last_iteration: 
            torch.autograd.set_grad_enabled(True)
            g = λ*softmin(ε, y,x, (log_α + f/ε)).detach()
            f = softmin(ε, x,y, (log_β + g/ε).detach())
                        
    # RSUOT cost
    SUOT_ε= torch.sum(α*f)+torch.sum(β*ρ*(1-torch.exp(-g/ρ)))

    return SUOT_ε,f.detach()

# RUOT Sinkhorn ########################################

def RUOT(x,y,ε,ρ,fg_init=None,nb_it=10,dev='cuda'): # g puis f 
    N= x.shape[0]
    M= y.shape[0]
    
    # no grad required for the optimisation part
    torch.autograd.set_grad_enabled(False)
    
    # Distributions
    α = 1 /N * torch.ones(x.shape[0], dtype=torch.float32).to(device=dev) # distribution uniforme α
    β = 1 /M * torch.ones(y.shape[0], dtype=torch.float32).to(device=dev) # distribution uniforme β
    log_α=torch.log(α)
    log_β=torch.log(β)
    
    # ε schedule
    ε_s=[ε]*nb_it
    last_iteration=len(ε_s)-1

    # initialise dual variables
    if fg_init==None:
        f = torch.zeros_like(α)
        g = torch.zeros_like(β)
    if fg_init!=None:
        f = fg_init[0]
        g = fg_init[1]
    
    # Optimisation of dual variables
    λ = 1 / (1 + ε / ρ)
    for i, ε in enumerate(ε_s):
        f,g= 0.5*(λ*softmin(ε, x, y, log_β + g/ε)+f), 0.5*(λ*softmin(ε, y, x, log_α + f/ε)+g)
        
        if i==last_iteration: # prepare gradient for autodiff
            torch.autograd.set_grad_enabled(True)
            g = λ*softmin(ε, y,x, (log_α + f/ε)).detach()
            f = λ*softmin(ε, x,y, (log_β + g/ε).detach())
                        
    # RUOT cost
    OT_ε= torch.sum(α*(ρ+ε)*(1-torch.exp(-f/ρ)))+torch.sum(β*ρ*(1-torch.exp(-g/ρ)))
    
    return OT_ε,[f.detach(),g.detach()]

