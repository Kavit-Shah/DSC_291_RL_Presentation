# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(layout_file="layouts/presentation.slides.json")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # On Function Approximation in Reinforcement Learning

    ## Optimism in the Face of Large State Spaces

    **Zhuoran Yang, Chi Jin, Zhaoran Wang, Mengdi Wang, Michael I. Jordan**

    *arXiv, 2021*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Big Question

    Imagine a robot dropped into an enormous maze with **millions of rooms**.

    - It cannot visit every room before making decisions.
    - So it must **generalize** from limited experience.
    - Core question of the paper:

    > Can we learn efficiently in huge state spaces using kernels or neural networks — and *prove* it?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why This Matters

    - Deep RL has major practical wins (AlphaGo, robotics, game playing).
    - But for **nonlinear function approximation**, theory lagged behind practice.
    - Earlier regret guarantees mainly covered:
      - Tabular RL (small finite state spaces), or
      - Linear function approximation.
    - **This paper**: first provably efficient framework for **kernel + neural-network** function approximation in RL.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Background: Episodic MDPs

    An episodic MDP is defined by $(\mathcal{S}, \mathcal{A}, H, \mathbb{P}, r)$:

    | Symbol | Meaning | Intuition |
    |--------|---------|-----------|
    | $\mathcal{S}$ | State space (possibly infinite) | All "situations" the agent can be in |
    | $\mathcal{A}$ | Finite action set | Choices available at each step |
    | $H$ | Horizon | Number of steps per episode |
    | $\pi = \{\pi_h\}$ | Policy | Decision rule: which action to take in each state |
    | $V_h^\pi(x)$ | Value function | Expected future reward from state $x$ at step $h$ |
    | $Q_h^\pi(x,a)$ | Q-function | Expected future reward after taking action $a$ in state $x$ |

    **Bellman optimality equation** (the recursive structure of optimal decisions):

    $$Q_h^*(x,a) = \bigl(r_h + \mathbb{P}_h V_{h+1}^*\bigr)(x,a)$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Exploration-Exploitation Dilemma

    - **Exploit**: pick actions that currently look best.
    - **Explore**: try uncertain states/actions that *might* be better.

    In huge state spaces, naive random exploration is hopeless — you'll never stumble on the good stuff by accident.

    We need **structured, uncertainty-aware exploration**: a principled way to decide *where* to explore.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Regret: Measuring Learning Speed

    $$\mathrm{Regret}(T) = \sum_{t=1}^{T}\bigl[V_1^*(x_1^t) - V_1^{\pi^t}(x_1^t)\bigr]$$

    - **Regret** = total reward lost compared to a perfect (omniscient) policy.
    - If regret grows **sublinearly** (e.g., $\widetilde{O}(\sqrt{T})$), then:
      - Average regret per episode $\to 0$
      - The algorithm is **converging to optimal** behavior.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Function Approximation: Why and What

    When $|\mathcal{S}|$ is huge or infinite, we can't store a Q-value for every state.

    **Solution**: approximate $Q^*(x,a)$ with a function from a class $\mathcal{F}$.

    | Approach | Pros | Cons |
    |----------|------|------|
    | **Linear**: $Q(x,a) \approx \phi(x,a)^\top\theta$ | Simple, well-understood theory | Often too restrictive |
    | **Kernel (RKHS)** | Rich, infinite-dimensional, analyzable | Computational cost |
    | **Neural networks** | Extremely expressive | Hard to analyze theoretically |

    **Key bridge**: Overparameterized (very wide) neural nets $\approx$ kernel methods via the **Neural Tangent Kernel (NTK)**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Core Idea: Optimism in the Face of Uncertainty

    > "When you're unsure about a state, assume it's *great* until proven otherwise."

    This is the **Upper Confidence Bound (UCB)** principle:

    - If the state actually IS great $\Rightarrow$ you benefit from going there early.
    - If it's NOT great $\Rightarrow$ you quickly learn this and won't go back.
    - **Either way, you make progress.** You never get stuck.

    This principle has been used in bandits and tabular RL for decades. This paper extends it to **nonlinear function classes** (kernels and neural nets).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Algorithm: Optimistic LSVI

    Repeat for each episode $t = 1, \dots, T$:

    ---

    **Step 1 -- Fit** (learn from past experience via least-squares regression):

    $$\hat{Q}_h^t = \arg\min_{f \in \mathcal{F}}\sum_{\tau=1}^{t-1}\bigl[r_h^\tau + V_{h+1}^t(x_{h+1}^\tau) - f(x_h^\tau, a_h^\tau)\bigr]^2 + \mathrm{pen}(f)$$

    **Step 2 -- Boost** (add an optimism bonus for uncertain state-actions):

    $$Q_h^t = \min\bigl\{\hat{Q}_h^t + \beta\, b_h^t,\; H-h+1\bigr\}^{+}$$

    **Step 3 -- Act** (pick the best-looking action under optimism):

    $$a_h^t = \arg\max_{a \in \mathcal{A}} Q_h^t(x_h^t, a)$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Bonus Function: Where the Magic Happens

    **Kernel setting (KOVI)**, for $z = (x, a)$:

    $$b_h^t(x,a) = \lambda^{-1/2}\Bigl[K(z,z) - k_h^t(z)^\top\bigl(K_h^t + \lambda I\bigr)^{-1}k_h^t(z)\Bigr]^{1/2}$$

    **Intuition**: this measures *"how different is this state-action from everything I've seen before?"*

    - **High bonus** = unfamiliar territory $\Rightarrow$ explore it!
    - **Low bonus** = well-visited region $\Rightarrow$ trust your Q-estimate.

    **Neural setting (NOVI)**: same idea, but uses gradient features $\phi(x,a; W)$ from the neural network as the "feature map."
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## KOVI vs NOVI: Two Instantiations

    |  | **KOVI** | **NOVI** |
    |---|---|---|
    | Function class | RKHS (kernel) | Overparameterized neural net |
    | Bonus computed from | Kernel Gram matrix $K_h^t$ | NTK gradient features $\phi(x,a;W)$ |
    | Theory | Exact (finite-sample kernel analysis) | Approximate (NTK linearization) |
    | Key connection | Kernel ridge regression | Wide-net limit behaves like a kernel |
    | Approximation error | None | Decays as network width $m \to \infty$ |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Main Result: Regret Bound

    $$\mathrm{Regret}(T) = \widetilde{O}\!\left(H^2\sqrt{T \cdot d_{\mathrm{eff}}}\right)$$

    | Term | Meaning | Intuition |
    |------|---------|-----------|
    | $T$ | Number of episodes | More experience $\Rightarrow$ better performance |
    | $H$ | Horizon length | Longer episodes $\Rightarrow$ harder to plan |
    | $d_{\mathrm{eff}}$ | Effective dimension of $\mathcal{F}$ | Simpler function class $\Rightarrow$ tighter bound |

    **The punchline**: No dependence on $|\mathcal{S}|$. Works even for **infinite** state spaces.

    Since regret grows as $\sqrt{T}$, the **average regret per episode** $\to 0$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Regret Under Different Function Classes

    | Function Class | Regret Bound |
    |---|---|
    | Finite spectrum (dim $\gamma$) | $\widetilde{O}(H^2\sqrt{\gamma^3 T})$ |
    | Exponential eigenvalue decay | $\widetilde{O}(H^2\sqrt{(\log T)^{3/\gamma}\, T})$ |
    | Polynomial eigenvalue decay | $\widetilde{O}(H^2\, T^{\kappa^* + \xi^* + 1/2})$ |
    | Overparameterized neural net | RKHS bound $+ \mathrm{poly}(T,H) \cdot m^{-1/12}$ |

    - **Smoother kernels** (faster eigenvalue decay) $\Rightarrow$ tighter bounds.
    - **Wider neural nets** ($m \to \infty$) $\Rightarrow$ approach kernel guarantees.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Proof Sketch (High Level)

    Three key steps:

    **1. Decompose regret** into three pieces:

    - TD (temporal-difference) approximation errors
    - Martingale noise (random fluctuations that average out)
    - Policy gap (non-positive because we act greedily)

    **2. Prove optimism**: the bonus is large enough that $Q_h^t \geq Q_h^*$ always holds.
    This ensures we never systematically under-explore.

    **3. Bound total uncertainty** via the maximal information gain $\Gamma_K$:

    - Each visit to a region **shrinks** the bonus there (self-normalization).
    - The total "optimism budget" across all episodes is bounded.
    - This is what gives us the $\sqrt{T}$ rate.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interactive Demo: Naive vs Optimistic Exploration

    **Environment**: 1D chain with 10 states, 2 actions (left / right), reward only at the rightmost state.

    | Agent | Strategy |
    |-------|----------|
    | **Naive** ($\varepsilon$-greedy) | Random exploration with probability $\varepsilon$ |
    | **Optimistic** (UCB-style) | Picks action by $Q(s,a) + \beta / \sqrt{N(s,a)+1}$ |

    This is a tabular toy — but the **same optimism principle** is exactly what the paper lifts to kernels and neural networks.

    Adjust the sliders below and click **Run simulation** to compare.
    """)
    return


@app.cell
def _(mo):
    episodes_slider = mo.ui.slider(
        start=50,
        stop=500,
        step=10,
        value=200,
        label="Episodes",
    )
    epsilon_slider = mo.ui.slider(
        start=0.01,
        stop=0.40,
        step=0.01,
        value=0.10,
        label="Naive epsilon",
    )
    beta_slider = mo.ui.slider(
        start=0.2,
        stop=4.0,
        step=0.1,
        value=1.8,
        label="Optimism beta",
    )
    run_button = mo.ui.run_button(label="Run simulation")
    return beta_slider, episodes_slider, epsilon_slider, run_button


@app.cell
def _(beta_slider, episodes_slider, epsilon_slider, mo, run_button):
    mo.vstack(
        [
            episodes_slider,
            epsilon_slider,
            beta_slider,
            run_button,
        ]
    )
    return


@app.cell
def _(beta_slider, episodes_slider, epsilon_slider, mo, np, plt, run_button):
    mo.stop(
        not run_button.value, mo.md("*Click **Run simulation** above to see results.*")
    )

    _n_states = 10
    _n_actions = 2
    _horizon = 2 * (_n_states - 1)
    _n_episodes = int(episodes_slider.value)
    _epsilon = float(epsilon_slider.value)
    _beta = float(beta_slider.value)
    _alpha = 0.35
    _gamma = 1.0
    _seed = 7

    def _transition(state, action):
        if action == 0:
            nxt = max(0, state - 1)
        else:
            nxt = min(_n_states - 1, state + 1)
        rew = 1.0 if nxt == _n_states - 1 else 0.0
        dn = nxt == _n_states - 1
        return nxt, rew, dn

    def _run_naive():
        rng = np.random.default_rng(_seed)
        qv = np.zeros((_n_states, _n_actions))
        regrets = np.zeros(_n_episodes)
        sv = np.zeros(_n_states)
        for ep in range(_n_episodes):
            s = 0
            ep_r = 0.0
            for _ in range(_horizon):
                sv[s] += 1
                if rng.random() < _epsilon:
                    a = int(rng.integers(_n_actions))
                else:
                    best = np.flatnonzero(qv[s] == qv[s].max())
                    a = int(rng.choice(best))
                ns, r, d = _transition(s, a)
                qv[s, a] += _alpha * (r + _gamma * np.max(qv[ns]) - qv[s, a])
                ep_r += r
                s = ns
                if d:
                    sv[s] += 1
                    break
            regrets[ep] = 1.0 - ep_r
        return np.cumsum(regrets), sv

    def _run_optimistic():
        rng = np.random.default_rng(_seed)
        qv = np.zeros((_n_states, _n_actions))
        vc = np.zeros((_n_states, _n_actions))
        regrets = np.zeros(_n_episodes)
        sv = np.zeros(_n_states)
        for ep in range(_n_episodes):
            s = 0
            ep_r = 0.0
            for _ in range(_horizon):
                sv[s] += 1
                bonus = _beta / np.sqrt(vc[s] + 1.0)
                scores = qv[s] + bonus
                best = np.flatnonzero(scores == scores.max())
                a = int(rng.choice(best))
                vc[s, a] += 1
                ns, r, d = _transition(s, a)
                qv[s, a] += _alpha * (r + _gamma * np.max(qv[ns]) - qv[s, a])
                ep_r += r
                s = ns
                if d:
                    sv[s] += 1
                    break
            regrets[ep] = 1.0 - ep_r
        return np.cumsum(regrets), sv

    _reg_naive, _vis_naive = _run_naive()
    _reg_opt, _vis_opt = _run_optimistic()

    _ckpts = min(8, _n_episodes)
    _ck_idx = np.linspace(0, _n_episodes - 1, _ckpts, dtype=int)
    _ck_labels = [str(int(i + 1)) for i in _ck_idx]

    _bw = 0.38
    _xr = np.arange(_ckpts)
    _xs = np.arange(_n_states)

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.8))

    _axes[0].bar(
        _xr - _bw / 2,
        _reg_naive[_ck_idx],
        width=_bw,
        color="#c44536",
        label=f"Naive eps-greedy (eps={_epsilon:.2f})",
    )
    _axes[0].bar(
        _xr + _bw / 2,
        _reg_opt[_ck_idx],
        width=_bw,
        color="#2d6a4f",
        label=f"Optimistic UCB (beta={_beta:.1f})",
    )
    _axes[0].set_xticks(_xr)
    _axes[0].set_xticklabels(_ck_labels)
    _axes[0].set_xlabel("Episode")
    _axes[0].set_ylabel("Cumulative regret")
    _axes[0].set_title("Cumulative Regret Over Time")
    _axes[0].legend(frameon=False)

    _axes[1].bar(
        _xs - _bw / 2, _vis_naive, width=_bw, color="#f4a261", label="Naive visits"
    )
    _axes[1].bar(
        _xs + _bw / 2, _vis_opt, width=_bw, color="#2a9d8f", label="Optimistic visits"
    )
    _axes[1].set_xticks(_xs)
    _axes[1].set_xlabel("State index (0=start, 9=goal)")
    _axes[1].set_ylabel("Visit count")
    _axes[1].set_title("State Visitation Distribution")
    _axes[1].legend(frameon=False)

    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(rf"""
    **Results after {_n_episodes} episodes**

    - Naive ($\varepsilon$-greedy) cumulative regret: **{float(_reg_naive[-1]):.1f}**
    - Optimistic (UCB-style) cumulative regret: **{float(_reg_opt[-1]):.1f}**

    Lower is better. Notice how optimism drives the agent to the goal state faster.
        """),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    1. **First provably efficient** RL framework with nonlinear function approximation (kernels + neural nets).
    2. The **optimism principle** (UCB) extends cleanly from tabular/linear to rich function classes.
    3. Regret depends on **function-class complexity** ($d_{\mathrm{eff}}$), **not** the size of the state space.
    4. **NTK theory** bridges wide neural nets to kernel-style guarantees.
    5. This paper **narrows the gap** between deep RL practice and RL theory.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Limitations and Open Questions

    **Assumptions**:

    - **Bellman completeness**: $\mathcal{F}$ must be "closed" under the Bellman operator — restrictive in practice.
    - Kernel eigenvalue decay rate must be known a priori.
    - NTK regime requires **very wide** networks (gap with practical architectures).

    **Computational cost**: Gram matrix inversion is $O(t^3)$ per step — expensive.

    **Open directions**:

    - Relax completeness / handle model misspecification.
    - Scale to practical deep-net widths.
    - Reduce per-step computation while preserving guarantees.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## References

    - Yang, Jin, Wang, Wang, Jordan (2021). *On Function Approximation in RL: Optimism in the Face of Large State Spaces.*
    - Jin et al. (2020). *Provably Efficient RL with Linear Function Approximation.*
    - Jacot, Gabriel, Hongler (2018). *Neural Tangent Kernel: Convergence and Generalization in Neural Networks.*
    - Srinivas et al. (2010). *Gaussian Process Optimization in the Bandit Setting (GP-UCB).*
    - Auer, Cesa-Bianchi, Fischer (2002). *Finite-time Analysis of the Multiarmed Bandit Problem.*

    ---

    *Thank you! Questions?*
    """)
    return


if __name__ == "__main__":
    app.run()
