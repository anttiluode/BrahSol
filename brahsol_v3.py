"""BrahSol v3: destructive reads, route addressing, and efference-copy controls.

All dynamics are deterministic synthetic TM-like dynamics; noisy READOUT ONLY.
No biological model-fitting, external neural data or neuronal intelligence claim.

Run: python brahsol_v3.py --out results_v3
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss

N = 8
U = 0.09
TAU_F = 30.0
TAU_D = 90.0
TAUS = 2.0 ** np.arange(1, 9)
LAM = np.exp(-1.0 / TAUS)
G_AUG = np.array([0.,0.,0.,0.,0.2,0.3,0.3,0.2])
MASKS = np.array([np.isin(range(N), list(c)) for c in itertools.combinations(range(N),4)], dtype=bool)
BINARY = np.array([[(j>>bit)&1 == 1 for j in range(N)] for bit in range(3)], dtype=bool)
Z, ZW = hermgauss(9)
Z = np.sqrt(2)*Z
ZW /= np.sqrt(np.pi)


def tm_step(u, r, mask=None, dt=1):
    """Exact decay + spike update; input arrays (H=8 hypotheses, E=8 routes)."""
    uu = U + (u - U)*np.exp(-dt/TAU_F)
    rr = 1. - (1.-r)*np.exp(-dt/TAU_D)
    if mask is None:
        return uu, rr, np.zeros(uu.shape)
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), uu.shape)
    uu = uu + U*(1.-uu)*mask
    output = uu*rr*mask
    rr = rr*(1.-uu*mask)
    return uu,rr,output


def initial_tm():
    u=np.full((N,N),U)
    r=np.ones((N,N))
    identity=np.eye(N,dtype=bool)
    for t in range(30):
        u,r,_=tm_step(u,r,identity if t in (0,2,4) else None)
    return u,r


def initial_linear():
    m=np.zeros((N,N,len(LAM)))
    identity=np.eye(N,dtype=bool)
    for t in range(30):
        pulse=identity.astype(float) if t in (0,2,4) else np.zeros((N,N))
        m=LAM[None,None,:]*m + pulse[:,:,None]
    return m


def linear_step(m,mask=None,dt=1):
    m=m*(LAM[None,None,:]**dt)
    if mask is None:
        return m,np.zeros((N,N))
    b=np.empty_like(m)
    b[:,:,0]=m[:,:,0]
    b[:,:,1:]=m[:,:,1:]-m[:,:,:-1]
    gain=1.+(b*G_AUG[None,None,:]).sum(-1)
    p=np.broadcast_to(np.asarray(mask,dtype=float), (N,N))
    return m+p[:,:,None],gain*p


def final_templates(kind,reads):
    """Same final diagnostic at tick 100; earlier global reads at 50,52,.."""
    if reads<0 or reads>24:
        raise ValueError("reads must be in 0..24 (before final read)")
    pulse_times=set(50+2*k for k in range(reads))
    if kind=="TM":
        u,r=initial_tm()
        # initial_tm ended after tick 29; iterate tick 30..99
        for t in range(30,100):
            u,r,_=tm_step(u,r,np.ones(N,bool) if t in pulse_times else None)
        hidden=np.concatenate((u,r),axis=1)
        u,r,out=tm_step(u,r,np.ones(N,bool))
    elif kind=="linear":
        m=initial_linear()
        for t in range(30,100):
            m,_=linear_step(m,np.ones(N,bool) if t in pulse_times else None)
        hidden=m.reshape(N,-1)
        m,out=linear_step(m,np.ones(N,bool))
    else: raise ValueError(kind)
    # rows: memory class label; cols: identity-preserving sensor readout
    return out,hidden


def diag_contrast(templates):
    # Label 0: written route0 versus route1 (unwritten).
    return float(templates[0,0]-templates[0,1])


def bayes_decoding(templates,sigma,n,seed):
    """Optimal Gaussian-template decoder under known model and noise variance."""
    rng=np.random.default_rng(seed)
    labels=rng.integers(N,size=n)
    obs=templates[labels]+rng.normal(0,sigma,size=(n,N))
    d2=np.sum((obs[:,None,:]-templates[None,:,:])**2,axis=-1)
    return float(np.mean(np.argmin(d2,axis=1)==labels))


def gate_read_wipe(n=12000,seed=77):
    records=[]
    # Equalize first diagnostic READ SNR across different model amplitudes;
    # use same normalized noise for subsequent reads within each model.
    for model in ("linear","TM"):
        base,_=final_templates(model,0)
        ref_contrast=abs(diag_contrast(base))
        sigma=ref_contrast/2.0
        base_hidden=final_templates(model,0)[1]
        base_sep=float(np.linalg.norm(base_hidden[0]-base_hidden[1]))
        for count in (0,1,3,5,8,12):
            t,h=final_templates(model,count)
            contrast=abs(diag_contrast(t))
            sep=float(np.linalg.norm(h[0]-h[1]))
            accuracy=bayes_decoding(t,sigma,n,seed+count)
            records.append(dict(model=model,previous_reads=count,final_tick=100,
                final_sensor_contrast=contrast,contrast_relative=contrast/ref_contrast,
                hidden_state_distance=sep,hidden_distance_relative=sep/base_sep,
                read_noise_sigma=sigma,read_accuracy=accuracy))
    return records


def information(means,posterior,sigma):
    """Quadrature estimate of I(hidden label; scalar response) per subset mask.

    means: [hypothesis=8, action=70]. Equal-variance Gaussian readout.
    All actions stimulate exactly four routes and therefore cost the same.
    """
    d=(means.T[:,:,None,None]+sigma*Z[None,None,:,None]
       -means.T[:,None,None,:])/sigma
    px=(np.exp(-.5*d*d)*posterior[None,None,None,:]).sum(-1)
    ratio=(-.5*Z[None,None,:]**2-np.log(np.maximum(px,1e-300)))
    return (posterior[None,:,None]*ratio*ZW[None,None,:]).sum((1,2))


def choose_mask(policy,u,r,logp,rng,t,sigma):
    if policy=="global":return np.ones(N,dtype=bool)
    if policy=="random":return MASKS[int(rng.integers(len(MASKS)))].copy()
    if policy=="fixed":return BINARY[t%3].copy()
    if policy=="active":
        uu=U+(u-U)*np.exp(-3/TAU_F)
        rr=1-(1-r)*np.exp(-3/TAU_D)
        next_u=uu+U*(1-uu)
        means=(next_u*rr)@MASKS.T
        posterior=np.exp(logp-np.max(logp));posterior/=posterior.sum()
        info=information(means,posterior,sigma)
        return MASKS[int(np.argmax(info))].copy()
    raise ValueError(policy)


def episode(seed,policy="active",steps=5,sigma=.012,threshold=None,cost=.025):
    """One label written by third party, later identified by limited scalar reads.

    Agent knows simulator equations and candidate memory patterns: model-based
    Bayesian experimental design, NOT a learned synaptic system. It never sees
    actual label or hidden state, only observations and own chosen masks.
    """
    rng=np.random.default_rng(seed)
    label=int(rng.integers(N))
    u,r=initial_tm()
    logp=np.zeros(N)
    masks=[];obs=[];reads=0
    for t in range(steps):
        posterior=np.exp(logp-logp.max());posterior/=posterior.sum()
        if threshold is not None and float(posterior.max())>=threshold:
            u,r,_=tm_step(u,r,None,dt=3)
            masks.append(None)
            obs.append(None)
            continue
        mask=choose_mask(policy,u,r,logp,rng,t,sigma)
        u,r,out=tm_step(u,r,mask,dt=3)
        means=out.sum(axis=1)
        noisy=float(means[label]+rng.normal(0,sigma))
        logp-=.5*((noisy-means)/sigma)**2
        masks.append(mask.copy());obs.append(noisy)
        reads+=1
    best=int(np.argmax(logp))
    return dict(label=label,prediction=best,correct=int(label==best),
         reads=reads,net_reward=int(label==best)-cost*reads,
         masks=masks,observations=obs)


def replay_posterior(masks,obs,sigma,history=None):
    """Diagnostic ablation: how much history the matched model retains.

    history=None -> full efference copy. history=0 -> suppress previous pings
    in the model (but still see current mask), 1 -> retain last one, etc.
    This is NOT the optimal marginalized no-copy Bayesian estimator.
    """
    logp=np.zeros(N)
    for t,(mask,y) in enumerate(zip(masks,obs)):
        if mask is None:continue
        u,r=initial_tm()
        for j in range(t):
            prior=masks[j] if history is None or j>=t-history else None
            u,r,_=tm_step(u,r,prior,dt=3)
        u,r,out=tm_step(u,r,mask,dt=3)
        means=out.sum(axis=1)
        logp-=.5*((y-means)/sigma)**2
    return np.exp(logp-logp.max())/np.exp(logp-logp.max()).sum()


def evaluate_addressability(seeds,steps=5,sigma=.012):
    result=[]
    for policy in ("active","fixed","random","global"):
        for s in seeds:
            rec=episode(int(s),policy,steps,sigma)
            result.append(dict(seed=int(s),policy=policy,steps=steps,sigma=sigma,
                correct=rec["correct"],reads=rec["reads"],
                net_reward=rec["net_reward"]))
    return result


def evaluate_efference(seeds,steps=5,sigma=.012):
    res=[]
    for s in seeds:
        rec=episode(int(s),"random",steps,sigma)
        for history in (None,0,1,2):
            p=replay_posterior(rec["masks"],rec["observations"],sigma,history)
            res.append(dict(seed=int(s),history="all" if history is None else str(history),
                correct=int(int(np.argmax(p))==rec["label"]),
                true_label_probability=float(p[rec["label"]])))
    return res


def train_stop_threshold(seeds,policy,sigma=.012,steps=5,cost=.025):
    """Reward-trained stop rule: select posterior-confidence threshold on TRAIN seeds.

    This uses a model-based decoder. Training only tunes a scalar stop threshold,
    not the route dynamics or the Bayesian classifier.
    """
    thresholds=[.5,.65,.8,.9,.95,1.0]
    scores={th:np.mean([episode(int(s),policy,steps,sigma,threshold=th,cost=cost)
                        ["net_reward"] for s in seeds]) for th in thresholds}
    return min(thresholds,key=lambda th:(-scores[th],th)),scores


def evaluate_stop(seeds,thresholds,steps=5,sigma=.012,cost=.025):
    res=[]
    for policy in ("active","fixed","random","global"):
        for s in seeds:
            rec=episode(int(s),policy,steps,sigma,threshold=thresholds[policy],cost=cost)
            res.append(dict(seed=int(s),policy=policy,threshold=thresholds[policy],
                correct=rec["correct"],reads=rec["reads"],reward=rec["net_reward"]))
    return res


def summary(records,keys,values):
    groups={}
    for r in records:
        k=tuple(r[f] for f in keys)
        groups.setdefault(k,[]).append(r)
    ans=[]
    for k,rows in sorted(groups.items(),key=lambda item:str(item[0])):
        d={f:v for f,v in zip(keys,k)}
        d["n"]=len(rows)
        for value in values:
            arr=np.array([r[value] for r in rows],dtype=float)
            d[value]={"mean":float(arr.mean()),"sd":float(arr.std(ddof=1)) if len(arr)>1 else 0.}
        ans.append(d)
    return ans


def save_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("results_v3"))
    p.add_argument("--seeds",type=int,default=300,
        help="Held-out test episodes per policy; training uses separate fixed seeds")
    p.add_argument("--sigma",type=float,default=.012)
    p.add_argument("--steps",type=int,default=5)
    args=p.parse_args()
    if args.seeds<10 or args.sigma<=0 or args.steps<1:
        p.error("seeds>=10, sigma>0, steps>=1 required")
    args.out.mkdir(parents=True,exist_ok=True)
    g0=gate_read_wipe()
    seedlist=np.arange(20000,20000+args.seeds)
    addressing=evaluate_addressability(seedlist,args.steps,args.sigma)
    efference=evaluate_efference(seedlist,args.steps,args.sigma)
    # Explicit train/test split; stop-rule thresholds tuned solely on train seeds
    trainseeds=np.arange(5000,5100)
    trained={}
    fit={}
    for policy in ("active","fixed","random","global"):
        trained[policy],fit[policy]=train_stop_threshold(trainseeds,policy,
                                             args.sigma,args.steps)
    stopping=evaluate_stop(seedlist,trained,args.steps,args.sigma)
    for name,rows in (("read_wipe",g0),("addressing",addressing),
                       ("efference",efference),("stopping",stopping)):
        save_csv(args.out/(name+".csv"),rows)
    out={"parameters":dict(seeds=args.seeds,sigma=args.sigma,steps=args.steps,
                   train_seeds="5000..5099",test_seeds="20000.."),
       "read_wipe":g0,
       "addressing":summary(addressing,["policy"],["correct","net_reward"]),
       "efference":summary(efference,["history"],["correct","true_label_probability"]),
       "stop_rule_training_scores":fit,
       "stop_thresholds":trained,
       "stopping":summary(stopping,["policy"],["correct","reads","reward"])}
    (args.out/"summary.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    for section in ("addressing","efference","stopping"):
        print("\n"+section)
        for r in out[section]:
            print(r)
    print("\nread-wipe contrast relative to silent control")
    for r in g0:
        print(r["model"],r["previous_reads"],round(r["contrast_relative"],4),round(r["read_accuracy"],3))
    print("Results:",args.out.resolve())

if __name__=="__main__":main()
