# Reinforcement Learning Details (SO-ARM101 Reach Task)

This document outlines the MDP formulation, state/action space specifications, reward function formulations, optimization hyperparameters, and the observation noise environments used to train policies for the SO-ARM101 robot arm.

---

## State and Action Spaces
- **Action Space ($6\text{D}$)**: Joint position targets for the 6 joints of the SO-ARM101 robot arm, normalized to $[-1, 1]$.
- **Control Frequency**: $30\text{ Hz}$ (Simulation $dt = 1/60\text{ s}$, decimation = 2).
- **Observation Space ($25\text{D}$)**:
  
  | Component | Description | Dimension |
  | :--- | :--- | :--- |
  | Joint positions ($\boldsymbol{q}_t$) | Relative joint angles of the 6 joints | 6D |
  | Joint velocities ($\dot{\boldsymbol{q}}_t$) | Relative joint velocities | 6D |
  | Target EE command ($\boldsymbol{g}_t$) | 3D position + 4D quaternion target orientation | 7D |
  | Previous action ($\boldsymbol{a}_{t-1}$) | Joint commands from the last control step | 6D |
  | **Total** | | **25D** |

---

## Reward Functions
The reward function is defined as a sum of position tracking, fine-grained position tracking, orientation tracking, and regularization penalties:

1. **End-Effector Position Tracking**: Encourages the gripper link $\boldsymbol{p}_{\mathrm{ee}}$ to reach the command target position $\boldsymbol{p}_{\mathrm{target}}$:
   $$R_{\mathrm{pos}} = - \|\boldsymbol{p}_{\mathrm{ee}} - \boldsymbol{p}_{\mathrm{target}}\|_2$$

2. **Fine-Grained Position Tracking**: A hyperbolic tangent penalty to provide dense gradient feedback near the target region:
   $$R_{\mathrm{fine\_pos}} = \tanh\left(-\frac{\|\boldsymbol{p}_{\mathrm{ee}} - \boldsymbol{p}_{\mathrm{target}}\|_2}{\sigma_{\mathrm{fine}}}\right)$$
   where $\sigma_{\mathrm{fine}} = 0.1\text{ m}$.

3. **End-Effector Orientation Tracking**: Quaternion error angle penalty:
   $$R_{\mathrm{ori}} = - \theta_{\mathrm{error}}$$
   where $\theta_{\mathrm{error}}$ is the angular rotation difference (in radians) between the gripper orientation and target orientation.

4. **Action Rate Penalty**: Regularization to smooth joint commands and prevent sudden jumps:
   $$R_{\mathrm{act\_rate}} = - \|\boldsymbol{a}_t - \boldsymbol{a}_{t-1}\|_2^2$$

5. **Joint Velocity Penalty**: Limits joint speed to avoid high motor strain:
   $$R_{\mathrm{joint\_vel}} = - \|\dot{\boldsymbol{q}}_t\|_2^2$$

### MDP Reward Configurations
| Reward Term | Formula | Initial Weight | Notes / Curriculum |
| :--- | :--- | :--- | :--- |
| $R_{\mathrm{pos}}$ | $-\|\boldsymbol{p}_{\mathrm{ee}} - \boldsymbol{p}_{\mathrm{target}}\|_2$ | $-0.2$ | None |
| $R_{\mathrm{fine\_pos}}$ | $\tanh(-\|\boldsymbol{p}_{\mathrm{ee}} - \boldsymbol{p}_{\mathrm{target}}\|_2 / 0.1)$ | $0.1$ | None |
| $R_{\mathrm{ori}}$ | $-\theta_{\mathrm{error}}$ | $-0.1$ | None |
| $R_{\mathrm{act\_rate}}$ | $-\|\boldsymbol{a}_t - \boldsymbol{a}_{t-1}\|_2^2$ | $-0.0001$ | Linearly decreases to $-0.005$ over 4500 training steps |
| $R_{\mathrm{joint\_vel}}$ | $-\|\dot{\boldsymbol{q}}_t\|_2^2$ | $-0.0001$ | Linearly decreases to $-0.001$ over 4500 training steps |

---

## PPO Hyperparameters
Polices are optimized using PyTorch via the RSL-RL PPO runner:

| Hyperparameter | Value |
| :--- | :--- |
| Actor Network | MLP: $[64, 64]$ hidden units, ELU activation |
| Critic Network | MLP: $[64, 64]$ hidden units, ELU activation |
| Initial Noise Scale ($\sigma_{\mathrm{init\_noise}}$) | $1.0$ |
| Parallel Environments | $4096$ |
| Rollout Horizon ($N$) | $24$ steps |
| Discount Factor ($\gamma$) | $0.99$ |
| GAE parameter ($\lambda$) | $0.95$ |
| Learning Epochs | $8$ epochs per iteration |
| Mini-batches | $4$ mini-batches per epoch |
| Learning Rate ($\alpha$) | $1.0 \times 10^{-3}$ |
| Learning Rate Schedule | Adaptive KL scheduler (target KL = $0.01$) |
| Max Gradient Norm | $1.0$ |
| Maximum Iterations | $1000$ |

---

## Observation Noise and Domain Randomization Scenarios
To make the trained policy robust for Sim-to-Real transfer, we inject time-correlated colored noise into the joint positions and velocities during training:

1. **No-Noise Baseline** (`Isaac-SO-ARM101-Reach-Sim2Real-Obs-No-Noise-v0`):
   - Training observation signals are clean and uncorrupted.

2. **Fixed Spectral Exponent Noise** (`Isaac-SO-ARM101-Reach-Sim2Real-Obs-Beta***-v0`):
   - Observation signals are corrupted by colored noise generated with a fixed spectral exponent $\beta$:
     $$\tilde{\boldsymbol{o}}_t^{(\mathrm{sens})} = \boldsymbol{o}_t^{(\mathrm{sens})} + \sigma_{\mathrm{obs}} \, \boldsymbol{\eta}_t^{(\beta)}$$
   - $\sigma_{\mathrm{obs}} = 0.02$.
   - $\beta \in \{0.25, 0.50, 0.75, 1.00, 1.25, 1.50\}$.
   - Noise sequence $\boldsymbol{\eta}_t^{(\beta)}$ is generated at the start of each episode using the Timmer-Koenig FFT algorithm to match the Power Spectral Density $S(f) \propto 1/f^{\beta}$.

3. **Spectral Exponent Randomization** (`Isaac-SO-ARM101-Reach-Sim2Real-Obs-Colored-Noise-v0`):
   - At each episode reset, each parallel environment $i$ samples a spectral exponent independently:
     $$\beta_i \sim \mathcal{U}(0.5, 1.5)$$
   - Noise $\boldsymbol{\eta}_t^{(\beta_i)}$ is generated with the sampled $\beta_i$.

### Summary of Noise Scenarios
| Environment ID | Noise Injected | Exponent Parameter $\beta$ | Noise Scale $\sigma_{\mathrm{obs}}$ |
| :--- | :---: | :--- | :--- |
| `Isaac-SO-ARM101-Reach-Sim2Real-Obs-No-Noise-v0` | No | -- | -- |
| `Isaac-SO-ARM101-Reach-Sim2Real-Obs-Beta0.25-v0` | Yes | $0.25$ (fixed) | $0.02$ |
| `Isaac-SO-ARM101-Reach-Sim2Real-Obs-Beta0.5-v0` | Yes | $0.50$ (fixed) | $0.02$ |
| `Isaac-SO-ARM101-Reach-Sim2Real-Obs-Beta0.75-v0` | Yes | $0.75$ (fixed) | $0.02$ |
| `Isaac-SO-ARM101-Reach-Sim2Real-Obs-Beta1.0-v0` | Yes | $1.00$ (fixed) | $0.02$ |
| `Isaac-SO-ARM101-Reach-Sim2Real-Obs-Beta1.25-v0` | Yes | $1.25$ (fixed) | $0.02$ |
| `Isaac-SO-ARM101-Reach-Sim2Real-Obs-Beta1.5-v0` | Yes | $1.50$ (fixed) | $0.02$ |
| `Isaac-SO-ARM101-Reach-Sim2Real-Obs-Colored-Noise-v0` | Yes | $\boldsymbol{\mathcal{U}(0.5, 1.5)}$ (randomized) | $0.02$ |
