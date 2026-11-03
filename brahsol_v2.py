"""BrahSol v2: does active probing track a *changing* neural medium?

All scores are obtained from actual runs of this self-contained NumPy toy.
Counterfactual reference probes use forks of ONE identical reference medium
for a given seed/time, so controllers face matched evaluation conditions.

No biological-neuron or causal-information-gain claims are made.
"""
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import numpy as np
from brahsol import NeuralMedium, ACTIONS, TARGET_ENERGY

POLICIES = ("random", "balanced", "homeo", "uncertainty", "active")


class ChangingMedium(NeuralMedium):
    """Leaky recurrent network with adaptation, Hebbian plasticity and a shock.

    Shock alters incoming synapses of ~1/3 of cells and reroutes input terminals. The medium's
    network and hidden state are never exposed to the controller. This is a
    SYNTHETIC reconfiguration, not a claim about actual biological rewiring.
    """
    def __init__(self, seed, plasticity=True, eta=0.016, shock=True):
        super().__init__(seed=seed, plasticity=plasticity, sensor_mode="four")
        self.eta = float(eta)
        self.shock_rng = np.random.default_rng(seed+474747)
        self.shock_enabled = bool(shock)
        self.shocked = False
        self.change_magnitude = 0.

    def change_connectivity(self):
        """A deterministic seed-dependent structural intervention, once."""
        if self.shocked or not self.shock_enabled:
            return
        old = self.W.copy()
        targets = self.shock_rng.choice(N_LOCAL, N_LOCAL//3, replace=False)
        # Make new incoming projections with a different pattern, holding the
        # current hidden state. The controller only sees the downstream return.
        for i in targets:
            random_row = self.shock_rng.normal(0, 0.11, N_LOCAL)
            self.W[i, :] = self.mask[i, :]*random_row
        # Rewire afferent terminals as well: some stimulus ports now access
        # different neural neighborhoods. This is what makes the shock
        # immediately observable, rather than a near-silent W perturbation.
        self.input = self.input[np.array([4,5,6,7,0,1,2,3])].copy()
        self.shocked = True
        self.change_magnitude = float(np.linalg.norm(self.W-old) /
                                      max(np.linalg.norm(old), 1e-12))

    def step(self, action: int):
        if not 0 <= action < ACTIONS:
            raise ValueError(action)
        old = self.h.copy()
        drive = self.W @ old + self.input[action] + self.bias - 0.55*self.adapt
        self.h = .60*old + .40*np.tanh(drive)
        self.adapt = .965*self.adapt + .035*np.maximum(self.h, 0)
        if self.plasticity:
            # Local pre-post Hebbian update with weight decay and fixed mask.
            # Unlike v1's eta=0.0008, this is deliberately stronger.
            self.W = np.clip((self.W + self.eta*(np.outer(self.h, old)
                       - 0.18*self.W))*self.mask, -0.5, 0.5)
            # Stable amplitude bound; avoids runaway while preserving learning.
            if self.step_count % 8 == 0:
                norm = float(np.linalg.norm(self.W, ord=2))
                if norm > 1.65:
                    self.W *= 1.65/norm
        out = self.readout@self.h
        out += self.rng.normal(0, 0.012, 4)
        self.last_y = out
        self.step_count += 1
        return out.copy()


N_LOCAL = 24


class AdaptiveRLS:
    """Action-conditioned one-step linear predictor with optional forgetting.

    The algorithm is identical under all action-selection policies. Forgetting
    rate gamma<1 makes the estimate weight recent observations more strongly.
    """
    def __init__(self, gamma=0.985, ridge=0.18):
        if not 0 < gamma <= 1:
            raise ValueError("gamma must lie in (0,1]")
        self.gamma = float(gamma)
        self.ridge = float(ridge)
        self.gram = np.tile(np.eye(5)[None,:,:]*ridge,(ACTIONS,1,1))
        self.cross = np.zeros((ACTIONS,5,4))
        self.weights = np.zeros((ACTIONS,5,4))
        self.counts = np.zeros(ACTIONS,dtype=int)

    @staticmethod
    def phi(y):
        return np.r_[1., np.asarray(y, dtype=float)]

    def predict(self,y,a):
        return self.phi(y)@self.weights[a]

    def leverage(self,y,a):
        x=self.phi(y)
        # Information surrogate; no Bayesian interpretation asserted.
        return max(0.,float(x@np.linalg.solve(self.gram[a],x)))

    def update(self,y,a,z):
        # Multiplying all port-specific statistics each tick discounts even
        # actions that have not been sampled recently: important for drift.
        self.gram = self.gamma*self.gram+(1-self.gamma)*self.ridge*np.eye(5)[None,:,:]
        self.cross *= self.gamma
        x=self.phi(y)
        self.gram[a] += np.outer(x,x)
        self.cross[a] += np.outer(x,z)
        self.weights[a]=np.linalg.solve(self.gram[a],self.cross[a])
        self.counts[a]+=1
        # Only action a was updated; other stale coefficients retain their old
        # value until that port is probed again. This is deliberate, measurable.


class Prober:
    def __init__(self,policy,seed,gamma=.985,beta=.25):
        if policy not in POLICIES: raise ValueError(policy)
        self.policy=policy
        self.model=AdaptiveRLS(gamma)
        self.rng=np.random.default_rng(seed+1717)
        self.beta=beta
        self.block=np.arange(ACTIONS)

    def choose(self,y,t):
        if self.policy=="random": return int(self.rng.integers(ACTIONS))
        if self.policy=="balanced" or t<3*ACTIONS:
            if t%ACTIONS==0: self.block=self.rng.permutation(ACTIONS)
            return int(self.block[t%ACTIONS])
        scores=[]
        for a in range(ACTIONS):
            m=self.model.predict(y,a)
            regulation=-50*(float(np.mean(m*m))-TARGET_ENERGY)**2
            gain=math.log1p(self.model.leverage(y,a))
            if self.policy=="homeo": score=regulation
            elif self.policy=="uncertainty": score=gain
            else: score=regulation+self.beta*gain
            scores.append(score)
        return int(np.argmax(scores))


@dataclass
class Receipt:
    seed:int
    policy:str
    gamma:float
    plasticity:bool
    shock:bool
    period:str
    ref_rmse:float
    own_prequential_rmse:float
    homeostasis_mse:float
    port_entropy:float
    synapse_drift:float
    shock_size:float


def paired_run(seed=0,steps=480,change_at=200,eval_every=8,
               plasticity=True,shock=True,gammas=(.985,1.),beta=.25):
    if steps <= change_at+90 or change_at<40: raise ValueError("Need >=90 post-change ticks")
    if eval_every<1: raise ValueError("eval_every must be >=1")
    agents={(p,g):Prober(p,seed,g,beta) for p in POLICIES for g in gammas}
    worlds={key:ChangingMedium(seed,plasticity,shock=shock) for key in agents}
    reference=ChangingMedium(seed,plasticity,shock=shock)
    reference_rng=np.random.default_rng(seed+2333)
    records={key:[] for key in agents}
    evals={key:[] for key in agents}
    choices={key:[] for key in agents}
    homeo={key:[] for key in agents}
    for t in range(steps):
        if t==change_at:
            reference.change_connectivity()
            for w in worlds.values(): w.change_connectivity()
        # Evaluate BEFORE any training update this tick. Counterfactual
        # outcomes from 8 action forks of the *same* reference state.
        if t%eval_every==0 or t in (change_at,change_at+1):
            y_ref=reference.last_y.copy()
            truth=np.stack([reference.clone().step(a) for a in range(ACTIONS)])
            for key,agent in agents.items():
                preds=np.stack([agent.model.predict(y_ref,a) for a in range(ACTIONS)])
                err=float(np.mean((preds-truth)**2))
                evals[key].append((t,err))
        for key,agent in agents.items():
            w=worlds[key]
            y=w.last_y.copy()
            a=agent.choose(y,t)
            pred=agent.model.predict(y,a)
            z=w.step(a)
            records[key].append(float(np.mean((pred-z)**2)))
            choices[key].append(a)
            homeo[key].append((float(np.mean(z*z))-TARGET_ENERGY)**2)
            agent.model.update(y,a,z)
        reference.step(int(reference_rng.integers(ACTIONS)))
    rows=[]
    for key,agent in agents.items():
        policy,gamma=key
        ports=np.bincount(choices[key],minlength=ACTIONS)
        probs=ports[ports>0]/sum(ports)
        entropy=float(-np.sum(probs*np.log(probs))/np.log(ACTIONS))
        for period,lo,hi in (("pre",max(0,change_at-80),change_at),
                             ("post_early",change_at,change_at+40),
                             ("post_late",change_at+80,steps)):
            errors=[e for t,e in evals[key] if lo<=t<hi]
            rows.append(Receipt(seed,policy,gamma,plasticity,shock,period,
                 float(np.sqrt(np.mean(errors))),
                 float(np.sqrt(np.mean(records[key][lo:hi]))),
                 float(np.mean(homeo[key][lo:hi])),entropy,
                 worlds[key].drift(),worlds[key].change_magnitude))
    return rows,evals


def aggregate(rows):
    groups={}
    for r in rows:
        k=(r.policy,r.gamma,r.plasticity,r.shock,r.period)
        groups.setdefault(k,[]).append(r)
    summary=[]
    for (policy,gamma,plasticity,shock,period),rs in sorted(groups.items()):
        def stat(name):
            v=np.array([getattr(r,name) for r in rs])
            return {"mean":round(float(v.mean()),6),"sd":round(float(v.std(ddof=1)),6) if len(v)>1 else 0.}
        summary.append({"policy":policy,"gamma":gamma,"plasticity":plasticity,
           "shock":shock,"period":period,"n":len(rs),
           "ref_rmse":stat("ref_rmse"),"own_rmse":stat("own_prequential_rmse"),
           "homeo_mse":stat("homeostasis_mse"),"entropy":stat("port_entropy"),
           "drift":stat("synapse_drift"),"shock_size":stat("shock_size")})
    return summary


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds",type=int,default=12)
    parser.add_argument("--steps",type=int,default=480)
    parser.add_argument("--change",type=int,default=200)
    parser.add_argument("--eval-every",type=int,default=8)
    parser.add_argument("--plasticity",choices=["on","off","both"],default="both")
    parser.add_argument("--shock",choices=["on","off","both"],default="both")
    parser.add_argument("--out",type=Path,default=Path("results_v2"))
    a=parser.parse_args()
    if a.seeds<1: parser.error("seeds must be >=1")
    rows=[]
    traces=[]
    for plastic in ([False,True] if a.plasticity=="both" else [a.plasticity=="on"]):
      for shock in ([False,True] if a.shock=="both" else [a.shock=="on"]):
        for seed in range(a.seeds):
            rs,ev=paired_run(seed,a.steps,a.change,a.eval_every,plastic,shock)
            rows.extend(rs)
            for (policy,gamma), observations in ev.items():
                for tick, mse in observations:
                    traces.append({"seed":seed,"policy":policy,"gamma":gamma,
                      "plasticity":plastic,"shock":shock,"tick":tick,
                      "reference_rmse":float(np.sqrt(mse))})
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/"runs.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(asdict(rows[0])));w.writeheader()
        w.writerows([asdict(r) for r in rows])
    with (a.out/"eval_trace.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(traces[0]));w.writeheader()
        w.writerows(traces)
    summ=aggregate(rows)
    (a.out/"summary.json").write_text(json.dumps(summ,indent=2),encoding="utf-8")
    print("policy       gamma plastic shock period       reference_RMSE    own_RMSE  homeo_MSE")
    for s in summ:
        print(f"{s['policy']:12} {s['gamma']:4.3f} {int(s['plasticity']):7} {int(s['shock']):5} {s['period']:12} "
              f"{s['ref_rmse']['mean']:9.4f} {s['own_rmse']['mean']:11.4f} {s['homeo_mse']['mean']:10.6f}")
    print("Wrote",a.out)


if __name__=="__main__":cli()
