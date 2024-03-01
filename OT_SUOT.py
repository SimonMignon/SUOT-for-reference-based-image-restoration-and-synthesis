import torch
from pykeops.torch import LazyTensor

# c-transform function #####################
def min_ot(x ,y , f):

    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])
    f_j = LazyTensor(f[None,:,None])
    
    D_ij = ((x_i - y_j) ** 2).sum(-1)
            
    rmv = D_ij - f_j
    amin = rmv.argmin(dim=1).view(-1)

    fmin=torch.sum((x-y[amin,:])**2,1)-f[amin]
    return fmin

# OT AGAA ##################################

def OT(x,y,nb_it,dev,g_init=None,lr=1):
    x_it=x.clone().detach()
    y_it=y.clone().detach()
    
    # initialise semi-dual variable
    if g_init==None:
        g= torch.zeros(y.shape[0],requires_grad=True,device=dev,dtype=torch.float32)
    else:
        g= g_init.requires_grad_(True)

    
    optim_g = torch.optim.ASGD([g], lr=lr, alpha=0.5, t0=1)
    
    # optimisation of g 
    for k in range(nb_it):
        optim_g.zero_grad()
        
        gc= min_ot(x_it ,y_it , g)
        E_ɛ = -torch.mean(gc)- torch.mean(g)
        
        E_ɛ.backward()
        optim_g.step()
        
    # OT cost 
    gc= min_ot(x ,y , g.detach())
    OT_ɛ= torch.mean(gc) + torch.mean(g.detach())
    
    return OT_ɛ,g.detach()

# SUOT AGAA ##################################

def SUOT(x,y,ρ,nb_it,dev,g_init=None,lr=0.1):
    x_it=x.clone().detach()
    y_it=y.clone().detach()
    
    # initialise semi-dual variable
    if g_init==None:
        g= torch.zeros(y.shape[0],requires_grad=True,device=dev,dtype=torch.float32)
    else:
        g= g_init.requires_grad_(True)

    
    optim_g = torch.optim.ASGD([g], lr=lr, alpha=0.5, t0=1)
    
    # optimisation of g 
    for k in range(nb_it):
        optim_g.zero_grad()

        gc= min_ot(x_it ,y_it , g)
        E_ɛ = -torch.mean(gc)- torch.mean(ρ*(1-torch.exp(-g/ρ)))
        
        E_ɛ.backward()
        optim_g.step()
        
    # SUOT cost 
    gc= min_ot(x ,y , g.detach())
    OT_ɛ= torch.mean(gc) + torch.mean(ρ*(1-torch.exp(-g.detach()/ρ)))

    return OT_ɛ,g.detach()