# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "altair==6.0.0",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(layout_file="layouts/presentation.slides.json")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return alt, go, make_subplots, mo, np, pd


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
    ## Intro: Problem Setting

    In modern RL, the state space is often too large to enumerate directly.

    - We cannot store a value for every state-action pair.
    - We must **generalize** from limited data using function approximation.
    - We still need exploration, and poor exploration can dominate performance.

    > Can we learn efficiently in huge state spaces using kernels or neural networks — and *prove* it?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Background (Single Slide): Prior Gap

    - Deep RL works well in practice, but theory for nonlinear approximation lagged behind.
    - Strong regret guarantees mostly covered:
      - Tabular RL (small finite state spaces), or
      - Linear approximation.
    - This paper closes that gap with a provably efficient framework for:
      - Kernel value functions,
      - Overparameterized neural-network value functions.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What This Paper Contributes

    1. Proposes **optimistic least-squares value iteration** with nonlinear approximation.
    2. Instantiates it as:
       - **KOVI** for RKHS/kernel value functions,
       - **NOVI** for overparameterized neural networks.
    3. Proves both are:
       - **Computationally efficient** (polynomial runtime),
       - **Statistically efficient** (sublinear regret).
    4. Gives bounds in terms of function-class complexity, not raw state count.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Paper Setup: Episodic MDP + Structural Assumption

    Setup used in the paper: episodic MDP $(\mathcal{S}, \mathcal{A}, H, \mathbb{P}, r)$.

    Three objects matter:

    - State $x$, action $a$, and reward $r$.
    - Horizon $H$ (steps per episode).
    - Value/Q functions for cumulative reward.

    Core paper assumption (informal):
    - Applying the Bellman operator keeps us inside a controlled function class.
    - This enables finite-time regret guarantees.

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
    ## Paper Challenge: Exploration with Function Approximation

    Why this is hard in this paper's setting:

    - Function approximation introduces estimation bias/variance tradeoffs.
    - RL additionally needs temporally-extended exploration.
    - Random exploration does not scale in large state spaces.
    - So the method must be both:
      - computationally tractable, and
      - statistically efficient.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Paper Objective: Regret Minimization

    $$\mathrm{Regret}(T) = \sum_{t=1}^{T}\bigl[V_1^*(x_1^t) - V_1^{\pi^t}(x_1^t)\bigr]$$

    - Regret = cumulative gap to the optimal policy.
    - Sublinear regret means learning improves over time.
    - Paper goal: prove sublinear regret with nonlinear function classes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Paper Function Classes: RKHS and Wide Neural Nets

    The paper studies two concrete choices for $\mathcal{F}$ to approximate $Q^*$:

    | Approach | Pros | Cons |
    |----------|------|------|
    | **Linear**: $Q(x,a) \approx \phi(x,a)^\top\theta$ | Simple, well-understood theory | Often too restrictive |
    | **Kernel (RKHS)** | Rich, infinite-dimensional, analyzable | Computational cost |
    | **Neural networks** | Extremely expressive | Hard to analyze theoretically |

    **Bridge used by the paper**: very wide neural nets can be analyzed via the induced NTK.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Paper Method Principle: Optimism (UCB)

    > If an action is uncertain, temporarily treat it as promising.

    This is the **Upper Confidence Bound (UCB)** principle:

    - If it is truly good, you discover it early.
    - If it is not good, uncertainty shrinks and you stop over-valuing it.
    - Net effect: efficient, directed exploration.

    This paper extends the UCB principle to nonlinear value-function classes (kernels and wide neural networks).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Algorithm: Optimistic LSVI

    Repeat for each episode $t = 1, \dots, T$:

    ---

    **Step 1 -- Fit** with least-squares regression:

    $$\hat{Q}_h^t = \arg\min_{f \in \mathcal{F}}\sum_{\tau=1}^{t-1}\bigl[r_h^\tau + V_{h+1}^t(x_{h+1}^\tau) - f(x_h^\tau, a_h^\tau)\bigr]^2 + \mathrm{pen}(f)$$

    **Step 2 -- Add bonus** for uncertain state-actions:

    $$Q_h^t = \min\bigl\{\hat{Q}_h^t + \beta\, b_h^t,\; H-h+1\bigr\}^{+}$$

    **Step 3 -- Act greedily** with the optimistic values:

    $$a_h^t = \arg\max_{a \in \mathcal{A}} Q_h^t(x_h^t, a)$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Bonus Function: Where the Magic Happens

    **Kernel setting (KOVI)**, for $z = (x, a)$:

    $$b_h^t(x,a) = \lambda^{-1/2}\Bigl[K(z,z) - k_h^t(z)^\top\bigl(K_h^t + \lambda I\bigr)^{-1}k_h^t(z)\Bigr]^{1/2}$$

    **Interpretation**: this is an uncertainty score.

    - **High bonus**: less data here $\Rightarrow$ explore.
    - **Low bonus**: well-covered region $\Rightarrow$ exploit learned values.

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

    **Core theorem template (both KOVI and NOVI):**

    $$\mathrm{Regret}(T) = \widetilde{O}\!\left(\delta_{\mathcal{F}}\,H^2\sqrt{T}\right)$$

    | Term | Meaning | Intuition |
    |------|---------|-----------|
    | $T$ | Number of episodes | More experience $\Rightarrow$ better performance |
    | $H$ | Horizon length | Longer episodes $\Rightarrow$ harder to plan |
    | $\delta_{\mathcal{F}}$ | Intrinsic complexity of function class | Encodes effective dimension + covering complexity |

    **The punchline**: No dependence on $|\mathcal{S}|$. Works even for **infinite** state spaces.

    Since regret grows as $\sqrt{T}$, the **average regret per episode** $\to 0$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Regret Under Different Spectral Regimes

    Corollaries in the paper specialize the general bound:

    | Function class assumption | Regret order |
    |---|---|
    | Finite spectrum (dim $\gamma$) | $\widetilde{O}(H^2\sqrt{\gamma^3 T})$ |
    | Exponential eigenvalue decay | $\widetilde{O}(H^2\sqrt{(\log T)^{3/\gamma}\, T})$ |
    | Polynomial eigenvalue decay | $\widetilde{O}(H^2\, T^{\kappa^* + \xi^* + 1/2})$ |
    | Overparameterized neural net | RKHS bound $+ \mathrm{poly}(T,H) \cdot m^{-1/12}$ |

    - **Smoother kernels** (faster eigenvalue decay) $\Rightarrow$ tighter bounds.
    - **Wider neural nets** ($m \to \infty$) $\Rightarrow$ approach kernel guarantees.
    - Next figure shows the **trend shape** of these rates (normalized, constants ignored).
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    _T = np.arange(50, 501, 10)
    _gamma = 8.0
    _finite = np.sqrt((_gamma**3) * _T)
    _expo = np.sqrt((_T) * (np.log(_T) ** (3.0 / _gamma)))
    _poly = _T**0.62

    _df = pd.DataFrame(
        {
            "T": np.concatenate([_T, _T, _T]),
            "Normalized regret proxy": np.concatenate(
                [_finite / _finite[0], _expo / _expo[0], _poly / _poly[0]]
            ),
            "Regime": np.repeat(
                ["Finite spectrum", "Exponential decay", "Polynomial decay"],
                repeats=len(_T),
            ),
        }
    )

    _chart = (
        alt.Chart(_df)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("T:Q", title="Episodes (T)"),
            y=alt.Y("Normalized regret proxy:Q", title="Normalized growth"),
            color=alt.Color(
                "Regime:N",
                scale=alt.Scale(
                    domain=["Finite spectrum", "Exponential decay", "Polynomial decay"],
                    range=["#355070", "#6d597a", "#b56576"],
                ),
            ),
        )
        .properties(width=760, height=330)
    )

    mo.vstack(
        [
            mo.md(
                r"""
    **Interpretation for class:** all curves are sublinear-like over this range, but smoother spectral structure gives better scaling.
    """
            ),
            _chart,
        ]
    )
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
    ## Optional Appendix Demo (Not a Paper Experiment)

    This short demo is only to build intuition for the optimism bonus used in the paper.

    **Environment**: 2D gridworld, sparse reward at a goal cell.

    | Agent | Exploration strategy |
    |-------|----------|
    | **Naive** ($\varepsilon$-greedy) | Random exploration with probability $\varepsilon$ |
    | **Optimistic** (UCB-style) | Chooses by $Q(s,a) + \beta / \sqrt{N(s,a)+1}$ |

    The paper's contribution is theoretical guarantees for nonlinear function classes (kernels + neural nets), not this toy benchmark.

    Adjust the sliders below and click **Run simulation** to compare.
    """)
    return


@app.cell
def _(mo):
    episodes_slider = mo.ui.slider(
        start=50,
        stop=400,
        step=10,
        value=220,
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
        value=1.6,
        label="Optimism beta",
    )
    run_button = mo.ui.run_button(label="Run simulation")
    mo.vstack([episodes_slider, epsilon_slider, beta_slider, run_button])
    return beta_slider, episodes_slider, epsilon_slider, run_button


@app.cell
def _(
    beta_slider,
    episodes_slider,
    epsilon_slider,
    go,
    make_subplots,
    mo,
    np,
    run_button,
):
    if not run_button.value:
        mo.stop(True, mo.md("*Click **Run simulation** to generate results.*"))

    _grid = 6
    _n_states = _grid * _grid
    _n_actions = 4
    _horizon = 4 * _grid
    _n_episodes = int(episodes_slider.value)
    _epsilon = float(epsilon_slider.value)
    _beta = float(beta_slider.value)
    _alpha = 0.35
    _gamma = 1.0
    _seed = 11
    _start = 0
    _goal = _n_states - 1

    def _to_xy(state):
        return state % _grid, state // _grid

    def _to_state(x, y):
        return y * _grid + x

    def _transition(state, action):
        x, y = _to_xy(state)
        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # down
            y = min(_grid - 1, y + 1)
        elif action == 2:  # left
            x = max(0, x - 1)
        else:  # right
            x = min(_grid - 1, x + 1)
        nxt = _to_state(x, y)
        rew = 1.0 if nxt == _goal else 0.0
        dn = nxt == _goal
        return nxt, rew, dn

    def _run_naive():
        rng = np.random.default_rng(_seed)
        qv = np.zeros((_n_states, _n_actions))
        regrets = np.zeros(_n_episodes)
        sv = np.zeros(_n_states)
        for ep in range(_n_episodes):
            s = _start
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
        rng = np.random.default_rng(_seed + 1)
        qv = np.zeros((_n_states, _n_actions))
        vc = np.zeros((_n_states, _n_actions))
        regrets = np.zeros(_n_episodes)
        sv = np.zeros(_n_states)
        for ep in range(_n_episodes):
            s = _start
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
    _episodes = np.arange(1, _n_episodes + 1)
    _vis_naive_grid = _vis_naive.reshape(_grid, _grid)
    _vis_opt_grid = _vis_opt.reshape(_grid, _grid)

    _fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "xy"}, {"type": "heatmap"}, {"type": "heatmap"}]],
        subplot_titles=(
            "Cumulative Regret",
            "Naive State Visits",
            "Optimistic State Visits",
        ),
        column_widths=[0.5, 0.25, 0.25],
    )
    _fig.add_trace(
        go.Scatter(
            x=_episodes,
            y=_reg_naive,
            mode="lines",
            name=f"Naive eps-greedy (eps={_epsilon:.2f})",
            line={"color": "#d1495b", "width": 3},
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=_episodes,
            y=_reg_opt,
            mode="lines",
            name=f"Optimistic UCB (beta={_beta:.1f})",
            line={"color": "#2a9d8f", "width": 3},
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Heatmap(
            z=_vis_naive_grid,
            colorscale="YlOrRd",
            showscale=False,
            name="Naive visits",
        ),
        row=1,
        col=2,
    )
    _fig.add_trace(
        go.Heatmap(
            z=_vis_opt_grid,
            colorscale="YlGnBu",
            showscale=False,
            name="Optimistic visits",
        ),
        row=1,
        col=3,
    )
    _fig.update_xaxes(title_text="Episode", row=1, col=1)
    _fig.update_yaxes(title_text="Cumulative regret", row=1, col=1)
    _fig.update_xaxes(title_text="x", row=1, col=2)
    _fig.update_xaxes(title_text="x", row=1, col=3)
    _fig.update_yaxes(title_text="y", autorange="reversed", row=1, col=2)
    _fig.update_yaxes(title_text="y", autorange="reversed", row=1, col=3)
    _fig.update_layout(
        height=460,
        width=1100,
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )

    _output = mo.vstack(
        [
            mo.md(
                rf"""
    **Results after {_n_episodes} episodes**

    - Naive ($\varepsilon$-greedy) cumulative regret: **{float(_reg_naive[-1]):.1f}**
    - Optimistic (UCB-style) cumulative regret: **{float(_reg_opt[-1]):.1f}**

    Lower is better. The optimistic agent usually reaches and revisits goal-directed regions faster.
                """
            ),
            _fig,
        ]
    )
    _output
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    1. **First provably efficient** RL framework with nonlinear function approximation (kernels + neural nets).
    2. The **optimism principle** (UCB) extends cleanly from tabular/linear to rich function classes.
    3. Regret depends on **function-class complexity** ($\delta_{\mathcal{F}}$), **not** the raw size of the state space.
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
