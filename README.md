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

These are repeated measurements from **one synthetic medium family**, not independent biological networks. Training policies can change hidden states and synapses, so end-of-training snapshots differ. Their test trajectories consequently differ too; this is not a controlled proof that one policy has a better representation in the same exact environment. A reference-world test and additional architectures are future work.

## Run

Requires Python 3.10+ and NumPy; no GPU or server.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python brahsol.py --seeds 20 --train 100 --test 120 --plasticity on --sensors four --out results_short
python brahsol.py --seeds 8 --train 650 --test 220 --plasticity on --sensors four --out results_long
python brahsol.py --seeds 20 --train 100 --test 120 --plasticity on --sensors four --shuffled-returns --out results_shuffled
python brahsol.py --seeds 5 --train 750 --test 240 --plasticity both --sensors both
```

Outputs: `runs.csv` (per seed) and `summary.json` (grouped mean metrics). `relative_skill = 1 - model_RMSE / persistence_RMSE`; action entropy is normalized to [0,1]; synaptic drift is relative Frobenius distance from the initial recurrent matrix. Homeostasis error is the mean squared deviation from the target squared sensor amplitude. Do not compare raw RMSE or homeostasis scores between the four-channel and summed-channel conditions without accounting for their different scales.

## What to falsify next

1. Evaluate every predictor on the **same** held-out reference medium and on separate fixed contextual states, rather than only each policy's own end state.
2. Vary observation budgets (1, 2, 4 ports), propagation delays, noise, and hidden network connectivity; include genuine unseen stimulation combinations.
3. Add controllable **probe locations**. Ask whether probe selection reduces observational equivalence / common nullspaces rather than simply improving regression.
4. Separate *neuron adaptation* from *synaptic plasticity* and quantify their individual effects.
5. Move from a linear return model to a genuinely predictive state estimator only if the simple model fails in demonstrable ways.

The target is not to obtain another rhythmic trace. It is to show a reduction in unseen-return prediction error **and** improved closed-loop regulation under matched observation and control budgets, without turning simple deterministic feedback into a story about sentience.

## Scope

This repository explores active sensing, online system identification, and feedback. Those are established research areas. The implementation is a self-contained toy designed to make sharp controls cheap. It does not establish a novel neuroscience mechanism or a theorem about neurons.