# BrahSol v3 — destructive reads, temporal self-history and route addresses

**Date:** 2026-11-03. **Scope:** controlled NumPy toy experiment. The experimental design was piloted first; the [confirmation plan](PREREG_V3.md) was committed before evaluating fresh synthetic seeds **20000–20399**. Thus the results are out-of-pilot confirmations, not an untouched first preregistration. Stop thresholds were fit on a separate set of seeds **5000–5099**.

## How the medium works

Eight independent hidden routes have Tsodyks–Markram-like synaptic utilization `u` (facilitation) and available resources `r` (depression). An external third party writes three pulses to exactly one unknown route (at ticks 0, 2, 4). Only **one scalar measurement** reaches the controller: the sum of responses from whichever route subset it chooses to stimulate. The controller does **not** see the label or internal `u,r` state.

Between pulses, `u` relaxes to baseline `U=0.09` with time constant `tau_f=30`, and `r` recovers toward 1 with `tau_d=90`. On a pulse, `u ← u+U(1-u)`, output is `u*r`, and `r ← r(1-u)`. Every probe therefore changes the medium. These dimensionless constants are engineered for a strong demonstration; they are *not* fitted to biological recordings.

For model-based route identification, each query activates **exactly four of eight** routes. There are 70 possible 4-route masks. The adaptive selector has the **correct synthetic model** and greedily maximizes a Gaussian-observation mutual-information estimate (nine-point Gauss–Hermite quadrature). This is an idealized observer, not a biological learner or proof that a neuron discovers the model.

## R0. Do repeated reads erase hidden memory?

The extra reads occur at ticks 50, 52, 54…; all conditions receive the same diagnostic at tick 100. No-extra-read is the reference. The linear Sihti comparison uses an eight-trace octave history plus its AUG readout, with identical broadcast pulses; its history differences are unchanged by common writes. Sensor noise is separately normalized so both models begin with comparable read-decoding accuracy. **Do not compare raw model amplitudes.**

| Earlier global reads | Linear contrast retained | TM contrast retained | TM hidden-state distance retained | TM final decoder accuracy |
|---:|---:|---:|---:|---:|
| 0 | 100% | 100% | 100% | 71.1% |
| 1 | 100% | 102.6% | 91.2% | 72.3% |
| 3 | 100% | 73.9% | 56.2% | 53.4% |
| 5 | 100% | 34.9% | 25.0% | 28.9% |
| 8 | 100% | 4.6% | 5.2% | 13.9% |
| 12 | 100% | **1.6%** | **1.6%** | **12.9%** |

The linear model's decoding accuracy remains about 71% for all interventions. Chance for 8 classes is 12.5%. TM's *hidden* states converge as well as its output, so this is more than a saturation/readout-only artifact. A single read can momentarily improve contrast before repeated reads collapse it.

## R1. Can one summed sensor infer a route's address?

400 distinct confirmation seeds (20000–20399), 5 queries, scalar Gaussian readout noise sigma=0.012. The global-broadcast control stimulates all 8 routes and is deliberately **not** equal-budget; each *selective* controller uses 4 stimulated routes per query, so active/fixed/random are equal-budget.

| Policy | Correct | Net reward (correct - 0.025 per read) |
|---|---:|---:|
| **Adaptive information query** | **99.00%** | **0.8650** |
| Fixed 3-bit subset code, repeated | 94.75% | 0.8225 |
| Random 4-of-8 subsets | 92.00% | 0.7950 |
| Global broadcast | 14.25% | 0.0175 |

Paired 400-seed differences: active minus random **+7.00 percentage points** (30 active-only correct, 2 random-only correct, 368 ties; paired SE ~1.37 pp); active minus fixed **+4.25 points** (21 active-only correct, 4 fixed-only correct, 375 ties; paired SE ~1.23 pp). In this toy family the adaptive policy improves addressing, but the fixed code already does remarkably well. Other noise levels and probe budgets were piloted; these confirmation figures refer strictly to the parameters above.

The broadcast response is mathematically identical under all eight labels, because the routes have identical parameters and exactly one is written. Selective masks break that symmetry. The controller is discovering an address **within the known hypothesis family**, not mapping an arbitrary unknown neural circuit.

## R2. How much does remembering one's own stimulation matter?

Using the *same* random-subset action/observation sequences as R1, rerun the Gaussian-template decoder with different amounts of earlier action history available to its forward model. This is an **ablation of a known-model observer**; the no-history decoder naïvely omits prior writes, rather than optimally marginalizing over them or learning a return-only representation. Therefore this result does **not** establish an information-theoretic necessity for an efference copy.

| Earlier self-pings retained | Decoding accuracy |
|---|---:|
| None | 37.0% |
| Most recent 1 | 53.5% |
| Most recent 2 | 59.5% |
| **All outgoing masks** | **92.0%** |

The same physical observations were replayed for every condition. The result demonstrates that correctly accounting for one's own back-action is vital **for this predictive decoder**. An optimized history-blind recurrent predictor remains an important missing control.

## R3. Is it worth choosing not to read?

Each controller trains only a simple *stopping rule*: select posterior confidence threshold from `{0.5,0.65,0.8,0.9,0.95,1.0}` using 100 independent training episodes. At test, if confidence exceeds that threshold, it makes no further read; otherwise it picks a route subset using its fixed/random/model-based policy. Net reward is terminal classification correctness minus **0.025** per performed read. The medium model and label templates are **supplied**, not learned.

| Policy with trained stop rule | Correct | Mean performed reads | Net reward |
|---|---:|---:|---:|
| **Adaptive, threshold 0.90** | **98.5%** | **3.61** | **0.8948** |
| Fixed binary, threshold 0.90 | 93.5% | 3.76 | 0.8410 |
| Random, threshold 0.95 | 91.75% | 4.20 | 0.8124 |
| Global broadcast, threshold 0.50 | 14.25% | 5.00 | 0.0175 |

Adaptive **always-read** obtained 99.0% using five reads, net reward 0.8650. The trained stop rule retains almost all accuracy while saving approximately **1.39 reads per episode** and increasing net reward to 0.8948. This is *learning when to stop asking* in a known synthetic world. It is **not** a learned state-space model or a newly established biological computation.

## Interpretation and limitations

- Positive result: selective probing breaks a severe observation symmetry even through one summed channel, despite probes modifying the hidden state. A controller that knows its own intervention history can model that back-action, and a simple reward-trained stopping rule reduces the cost of repeated reads.
- Physical distinction: common writes cancel in the difference of linear residue traces, but nonlinear facilitation/depletion causes state convergence; preserving useful information can require avoiding unnecessary queries.
- **No claim of neuron-level intelligence.** The model-based controller is given correct dynamics, noise, label templates and the finite eight-label hypothesis space. Only its one-number stopping threshold is trained from outcomes.
- All eight routes are independent and identically parameterized. More challenging tests should add recurrent between-route connections, heterogeneous delays and weights, unannounced third-party writes *during* interrogation, and unknown model parameters.
- Next **hard control**: compare exact-history decoding against an optimal history-marginalizing observer and a trained return-only recurrent estimator. Stop rules should be tested under changes of synaptic physics and to stimuli that were never used for training.
- In the fixed five-query experiment, different policies create different hidden states, so each observer explicitly updates its model with its own actions. The R2 ablation reuses the *same exact actions and returns* to isolate history accounting.

## Reproduce

```bash
python -m pip install numpy
python -m unittest discover -s tests -v
python brahsol_v3.py --seeds 400 --steps 5 --sigma 0.012 --out results_v3
```

Outputs: `read_wipe.csv`, `addressing.csv`, `efference.csv`, `stopping.csv`, `summary.json`. The full confirmation summary is also committed in `results_v3/summary.json`. The pre-confirmation design is in [PREREG_V3.md](PREREG_V3.md). All claims here are **in silico**.