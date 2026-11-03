# BrahSol — learning a medium by pinging it

**Question:** Can a bounded controller learn something useful about an unseen, *changing* neural medium from the returns to its own stimulation, without reading the medium's internal state?

This is the Sol branch of the November 3, 2026 Perception Laboratory / Geometric Neuron discussion. The parallel [BrahClaude](https://github.com/anttiluode/BrahClaude) branch is separate. Historical starting point: [GeometricNeuronOriginReview](https://github.com/anttiluode/GeometricNeuronOriginReview), especially `ecg.json` and its homeostatic checkerboard loop.

## Main distinction

The historical checkerboard is a fixed geometric transformation. This experiment replaces it with **24 hidden recurrent units**, leaky activation, an adapting internal state, optional slow Hebbian weight changes, and eight spatially distinct stimulus ports. Four fixed sensors provide a compressed return. The controller cannot inspect hidden states or synapses.

A regularized recursive least-squares model learns each port's one-step return map:

`y[t+1] ≈ M[action] · [1, y[t]]`

A controller can select the next ping to minimize homeostatic sensor-energy deviation, maximize an **uncertainty surrogate** (RLS covariance leverage), or use baseline schedules. The active policy combines the first two. This is *not* a mechanistic model of a dendritic tuft, a genuine biological neuron, or calibrated Bayesian information gain.

## Experiments and controls

All controllers get exactly the same number of pings and use the same predictor structure.

- **random:** independent uniform port choice
- **balanced:** randomly shuffled groups of eight ports, ensuring equal counts without temporal periodicity
- **sweep:** fixed 0,1,…,7 port sequence; a deliberately dangerous time-correlated control
- **homeo:** greedy predicted sensor-energy regulation, with initial exploration
- **active:** homeostasis plus exploration by covariance leverage, with the same initial exploration

Additional interventions: disable hidden synaptic plasticity; replace causal training returns with randomly drawn historical returns; replace four separate sensor values with their scalar sum. Test actions are freshly sampled and independent of the training actions. The predictor is **frozen** for the test.

### First receipts — toy model, not a neuroscience finding

Test return prediction error (RMSE, lower is better), **four sensors and synaptic plasticity enabled**:

| Policy | 20 seeds, 100 training pings | 8 seeds, 650 training pings |
|---|---:|---:|
| active | **0.0258** | 0.0164 |
| random | 0.0316 | 0.0165 |
| balanced | 0.0335 | 0.0177 |
| homeo | 0.0582 | 0.0529 |
| sweep | 0.0931 | 0.0958 |

At **100 pings**, active beat random and balanced on all 20 paired random seeds. At **650 pings**, active and random were virtually tied. Both matter: *adaptive probing helps when the interaction budget is small in this particular synthetic world, but has no durable advantage against uniform random probing once data are abundant.*

Controls at 100 pings / 20 seeds:

- Train with unrelated historical returns instead of correct action-return pairs: active RMSE **0.1085**, versus 0.0258 normally. The predictive relation matters.
- Turn off synaptic plasticity: active RMSE **0.0257**, essentially unchanged. **Therefore this pilot does not show that plasticity causes the advantage.**
- Fixed cyclic sweep performs particularly badly because action identity is locked to temporal phase during training, while test actions are random. Its failure is *not* evidence of active intelligence.
- Homeo strongly lowers sensor-energy deviation but harms held-out prediction in this configuration. Staying stable and learning a world are different objectives.

These are repeated measurements from **one synthetic medium family**, not independent biological networks. For the table above, training policies can change hidden states and synapses, so their test states differ. **We also ran a shared-world control below**: the same reference medium and the same held-out test actions for every policy. It confirmed the short-budget result without proving generality across network architectures.

## Run

Requires Python 3.10+ and NumPy; no GPU or server.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python brahsol.py --seeds 20 --train 100 --test 120 --plasticity on --sensors four --out results_short
python brahsol.py --seeds 8 --train 650 --test 220 --plasticity on --sensors four --out results_long
python brahsol.py --seeds 20 --train 100 --test 120 --plasticity on --sensors four --shuffled-returns --out results_shuffled
python brahsol.py --seeds 5 --train 750 --test 240 --plasticity both --sensors both
python shared_probe.py --seeds 20 --train 100 --test 120
python shared_probe.py --seeds 8 --train 650 --test 220
```

Outputs: `runs.csv` (per seed) and `summary.json` (grouped mean metrics). `relative_skill = 1 - model_RMSE / persistence_RMSE`; action entropy is normalized to [0,1]; synaptic drift is relative Frobenius distance from the initial recurrent matrix. Homeostasis error is the mean squared deviation from the target squared sensor amplitude. Do not compare raw RMSE or homeostasis scores between the four-channel and summed-channel conditions without accounting for their different scales.

## What to falsify next

1. Extend the **common-world** control to separately fixed contextual states, new mediums, and held-out network architectures; the initial same-world test is in `shared_probe.py`.
2. Vary observation budgets (1, 2, 4 ports), propagation delays, noise, and hidden network connectivity; include genuine unseen stimulation combinations.
3. Add controllable **probe locations**. Ask whether probe selection reduces observational equivalence / common nullspaces rather than simply improving regression.
4. Separate *neuron adaptation* from *synaptic plasticity* and quantify their individual effects.
5. Move from a linear return model to a genuinely predictive state estimator only if the simple model fails in demonstrable ways.

The target is not to obtain another rhythmic trace. It is to show a reduction in unseen-return prediction error **and** improved closed-loop regulation under matched observation and control budgets, without turning simple deterministic feedback into a story about sentience.

## Scope

This repository explores active sensing, online system identification, and feedback. Those are established research areas. The implementation is a self-contained toy designed to make sharp controls cheap. It does not establish a novel neuroscience mechanism or a theorem about neurons.
## Stronger control: identical evaluation world

`shared_probe.py` trains separate controllers on their own ping histories, then evaluates their frozen predictors on **exactly the same reference medium, its hidden state, and its independent random held-out actions** for each seed. This eliminates the evaluation-world confound (not the differences in training data).

| Policy | Shared-world RMSE: 20 seeds / 100 pings | Shared-world RMSE: 8 seeds / 650 pings |
|---|---:|---:|
| active | **0.02571** | **0.01631** |
| random | 0.03146 | 0.01654 |
| balanced | 0.03394 | 0.01831 |
| homeo | 0.05699 | 0.05183 |
| sweep | 0.09314 | 0.09546 |

For 100 pings, active beat random and balanced in **20/20 paired seeds** under shared-world evaluation. At 650 pings it only narrowly beat random (6/8 seeds). The common-world persistence baseline was about 0.1056 at 100 pings.

This supports a **sample-efficiency effect in the specified artificial environment**, not neural-level system identification or a biological learning mechanism. The active selector's `log(1+xᵀPx)` bonus chiefly encourages informative coverage; it is not an optimal experimental-design solution.

Most revealing negative control: disabling the medium's Hebbian synaptic plasticity did not change the short-budget active advantage (RMSE 0.0257 versus 0.0258 with plasticity). The phenomenon so far depends on interaction, state and sampling, **not demonstrably on synaptic learning**. More heterogeneous and changing media are needed to test that conjecture.


---

## V2: Can a learned ping-response model recover after the medium changes?

[**Full v2 results and controls**](RESULTS_V2.md) · [v2 implementation](brahsol_v2.py) · [v2 tests](tests/test_v2.py)

V2 deliberately changes synapses at tick 170, including which neural neighborhood receives each stimulation port. A controller now has to revise what a given ping means. It compares five stimulation policies with and without exponential forgetting and repeats everything with and without hidden Hebbian plasticity and without the engineered shock.

**Main receipt (12 seeds; synaptic learning enabled; shock at tick 170; late post-change):**

| Active + forgetting | Random + forgetting | Active without forgetting |
|---:|---:|---:|
| **0.0483 RMSE** | 0.0561 RMSE | 0.1008 RMSE |

Active with forgetting wins against random with forgetting on **11/12 paired seeds** in the late recovery window, but its benefit right after the shock is minimal. **Forgetting, not the active action selector, supplies the main recovery effect.** Almost the same behavior occurs with hidden Hebbian plasticity disabled, so this result **does not establish that a controller learns upstream neural plasticity**.

When no engineered shock occurs, not forgetting works better. This is a useful negative control: a flexible controller has to choose *when* to forget. Explicit changepoint detection remains future work.

Run from the repository root:

```bash
python -m unittest discover -s tests -v
python brahsol_v2.py --seeds 12 --steps 440 --change 170 --eval-every 12 --plasticity both --shock both --out results_v2
```

This leaves the v1 experiment intact. No biological apical-tuft mechanism is being claimed.


---

## V3: Destructive reading, efference copy, and addresses through one scalar return

[**Full v3 results**](RESULTS_V3.md) · [confirmation plan](PREREG_V3.md) · [source](brahsol_v3.py) · [11 tests](tests/test_v3.py) · [machine-readable summary](results_v3/summary.json)

Building on BrahClaude's temporal-Sihti route experiments, v3 asks what happens when reading a hidden synaptic state **changes that state**. It uses eight TM-like facilitation/depression routes with one summed sensory return and explicit Bayesian hypotheses for the eight possible written routes. The adaptive query strategy is given the **correct toy synapse model**; it is not a biological neural learner.

- **Destructive read:** after 12 earlier global pings, TM-like route-class contrast falls to **1.6%** of a no-extra-read control at the same final observation time. The linear Sihti comparator retains 100%. TM hidden-state distance also falls to 1.6%, separating actual state convergence from mere sensor saturation.
- **Addressability:** with 5 reads, 400 fresh confirmation seeds and a single summed observation, adaptive selective probes identify the written route **99.0%**, fixed three-bit code 94.75%, random four-route subsets 92.0%, and global broadcast 14.25% (12.5% chance). Adaptive, fixed and random activate exactly four routes per read.
- **Own-ping history:** decoding the *same random-subset recordings* with a complete action history gives 92.0% correct; truncating all earlier actions in that **specific predictor** gives 37.0%. This is a model ablation, not a proof an optimal history-blind estimator cannot recover the state.
- **Learned stopping:** an active policy with a stop threshold chosen from reward on separate training seeds makes **3.61 reads** on average, achieves 98.5% accuracy and scores 0.8948 net reward (correctness minus 0.025 per read). Always reading five times scores 0.8650.

**Interpretation:** selective addressing, back-action modeling and knowing when to remain silent can be useful in this engineered model. The physics and hypothesis templates are supplied to the controller; only the stop threshold is trained. This does **not** yet show a neuron learning its unknown connections, the synapse model, or an apical-tuft mechanism.

Run the 400-seed confirmation locally:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python brahsol_v3.py --seeds 400 --steps 5 --sigma 0.012 --out results_v3
```

The design was tuned in an exploratory pilot; `PREREG_V3.md` freezes the final confirmation on separate seeds 20000–20399 before those final evaluations.
