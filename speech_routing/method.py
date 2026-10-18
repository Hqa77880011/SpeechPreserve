import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;self.filters=nn.ParameterList([nn.Parameter(torch.ones(k)/k) for k in c.get('taps',[3,9,17])]);j=len(self.filters)
        self.risk=nn.Sequential(nn.Linear(4,32),nn.SiLU(),nn.Linear(32,3*j));self.j=j
    def forward(self,b):
        x=b['x'].float();n,time,frame=x.shape;queue=x.new_zeros(n);debt=x.new_zeros(n);previous=torch.full((n,),-1,dtype=torch.long,device=x.device);outputs=[];actions=[];risks=[];all_actions=[];prediction=[];queues=[];completion=[]
        latency=torch.tensor(self.c.get('latency_ms',[.2,.5,1.]),device=x.device);cost=torch.tensor([len(k) for k in self.filters],device=x.device).float()
        for t in range(time):
            context=x[:,:t+1].reshape(n,1,-1);branches=torch.stack([F.conv1d(F.pad(context,(len(k)-1,0)),k[None,None])[:,:,-frame:].squeeze(1) for k in self.filters],1)
            feature=torch.stack([x[:,t].mean(-1),x[:,t].std(-1,unbiased=False),queue,debt],-1);raw=self.risk(feature).reshape(n,3,self.j);quality,damage,margin=raw[:,0],F.softplus(raw[:,1]),F.softplus(raw[:,2]);upper=damage+margin
            switch=(torch.arange(self.j,device=x.device)[None]!=previous[:,None])
            duration_by_action=latency[None]+switch*self.c.get('switch_latency_ms',0.)
            memory=x.new_tensor(self.c.get('memory_kib',[0.]*self.j))
            feasible=(queue[:,None]+duration_by_action<=self.c.get('hop_ms',10.))&(queue[:,None]+duration_by_action<=self.c.get('deadline_ms',20.))
            feasible=feasible&(memory[None]<=self.c.get('memory_budget_kib',float('inf')))
            if self.c['variant']!='no_preservation':feasible=feasible&(upper<=self.c.get('damage_limit',.2))
            utility=quality-.01*debt[:,None]*cost
            if self.c['variant']=='no_debt':utility=quality
            if self.c['variant']!='no_switch_cost':utility=utility-.01*(torch.arange(self.j,device=x.device)[None]!=previous[:,None])
            choice=utility.masked_fill(~feasible,-1e9).argmax(-1);fallback=~feasible.any(-1)
            if self.c['variant']=='baseline':choice=torch.full_like(choice,self.j-1);fallback=torch.zeros_like(fallback)
            selected=branches[torch.arange(n,device=x.device),choice];selected=torch.where(fallback[:,None],x[:,t],selected)
            charged=torch.where(fallback,0.,cost[choice]);duration=torch.where(fallback,0.,duration_by_action[torch.arange(n,device=x.device),choice]);completion.append(queue+duration);queue=F.relu(queue+duration-self.c.get('hop_ms',10.));debt=F.relu(debt+charged-self.c.get('compute_budget',9.));previous=torch.where(fallback,-1,choice)
            outputs.append(selected);actions.append(torch.where(fallback,-1,choice));risks.append(upper);all_actions.append(branches);prediction.append(raw);queues.append(queue)
        return {'prediction':torch.stack(outputs,1),'action':torch.stack(actions,1),'damage_bound':torch.stack(risks,1),'branches':torch.stack(all_actions,1),'risk_prediction':torch.stack(prediction,1),'queue_ms':torch.stack(queues,1),'completion_ms':torch.stack(completion,1)}
    def objective(self,b,p):
        target=b['y'].float();branch=p['branches'];reference=target[:,:,None];energy=reference.square().sum(-1).clamp_min(1e-8);gain=(branch*reference).sum(-1)/energy
        damage=(gain-1).square()+((branch-gain[:,:,:,None]*reference).square().sum(-1)/energy)
        raw=p['risk_prediction'];upper=F.softplus(raw[:,:,1])+F.softplus(raw[:,:,2]);residual=damage.detach()-upper
        return F.mse_loss(branch,reference.expand_as(branch))+F.mse_loss(F.softplus(raw[:,:,1]),damage.detach())+torch.maximum(.9*residual,-.1*residual).mean()+F.mse_loss(raw[:,:,0],-damage.detach())


def demo(seed,n,c):
    rng=np.random.default_rng(seed);time=np.arange(16*80)/8000;clean=np.sin(2*np.pi*220*time)[None].repeat(n,0).reshape(n,16,80).astype('float32');return {'x':clean+rng.normal(0,.2,clean.shape).astype('float32'),'y':clean,'group':np.arange(n)}


def measure(p,d,c):
    x=p['prediction'];y=d['y'];gain=(x*y).sum((1,2))/np.maximum((y*y).sum((1,2)),1e-8);error=x-gain[:,None,None]*y
    sisdr=10*np.log10(np.maximum((gain[:,None,None]*y).reshape(len(y),-1).__pow__(2).sum(1),1e-8)/np.maximum((error**2).sum((1,2)),1e-8))
    return {'si_sdr_db':float(sisdr.mean()),'speech_gain_error':float(np.mean((gain-1)**2)),'bypass_fraction':float(np.mean(p['action']<0)),'deadline_violation':float(np.mean(p['completion_ms']>c.get('deadline_ms',20.))),'switch_fraction':float(np.mean(np.diff(p['action'],axis=1)!=0))}

def execute(args):
    from .runtime import execute as execute_backends
    execute_backends(args)
