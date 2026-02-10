## Slide 9: Method Principle - Optimism (UCB)
Now we get to the core idea that makes everything work in this paper: optimism in the face of uncertainty, or UCB. The intuition is simple. If I am uncertain about a state-action pair, I temporarily treat it as promising. If it is actually good, I discover it early. If it is not good, uncertainty shrinks and I stop over-valuing it. This gives directed exploration instead of random wandering, which is essential in large or continuous state spaces. The contribution here is extending that UCB principle to nonlinear function classes, where the usual tabular or linear arguments no longer apply.

## Slide 10: Algorithm - Optimistic LSVI
With that principle in mind, the algorithm is optimistic least-squares value iteration. Each episode, and for each time step, it does three things. First it fits a Q-function to Bellman targets using least-squares regression over the data collected so far. This gives a baseline estimate. Second, it adds an uncertainty bonus that depends on how confident we are at each state-action. Third, it acts greedily with respect to the optimistic values. So the policy is always the greedy one, but the optimism term steers it toward uncertain regions.

## Slide 11: Bonus Function
The bonus is where the technical innovation sits. In the kernel setting, the bonus is the predictive uncertainty from kernel ridge regression. The formula uses the kernel Gram matrix, and the bonus is large when a point is poorly covered by data, and small when it is well covered. In the neural network setting, the same idea is implemented through gradient features of a wide net. Those gradients act like a feature map for the neural tangent kernel, so the bonus is still an uncertainty score, just computed with the network’s induced kernel.

## Slide 12: KOVI vs NOVI
This gives two concrete instantiations. KOVI is the exact kernel method with finite-sample guarantees. NOVI is the neural version, where wide-network behavior approximates a kernel and the error shrinks as width grows. KOVI is cleaner theoretically, NOVI is closer to modern deep RL practice, and both are framed by the same optimistic LSVI template.

## Slide 13: Main Result - Regret Bound
The main theorem says the regret scales like $\widetilde{O}(\delta_{\mathcal{F}} H^2 \sqrt{T})$. The key message is what the bound depends on: the intrinsic complexity of the function class, not the size of the state space. That means the method can handle infinite or continuous state spaces as long as the function class is controlled. Because the growth is $\sqrt{T}$, the average regret per episode goes to zero.

## Slide 14: Spectral Regimes
The paper then refines the bound under different spectral assumptions about the kernel. If the spectrum is effectively finite, the regret has a clean polynomial dependence on the effective dimension. If eigenvalues decay exponentially, the bound is close to $\sqrt{T}$ up to log factors. If eigenvalues decay polynomially, the rate is slower but still sublinear. In the neural network case, you get the kernel bound plus an extra approximation term that shrinks with width. So smoother kernels and wider networks both help.

## Slide 15: Trend Figure
This figure visualizes those differences. All curves are sublinear in this range, but smoother spectral structure yields noticeably better scaling. The point of the slide is qualitative: the geometry of the function class drives the learning rate, and you can see that in these normalized trend lines.

## Slide 16: Proof Sketch
At a high level, the proof decomposes regret into three parts: temporal-difference approximation error, martingale noise, and a policy gap. The policy gap is non-positive because the agent acts greedily with optimistic values. The key technical lemma is optimism: the constructed Q-function upper bounds the true optimal Q-function. Finally, the total uncertainty across all episodes is bounded by the maximal information gain of the kernel, which gives the overall $\sqrt{T}$ behavior.

## Slide 17: Optional Appendix Demo
This is a short intuition demo, not a paper experiment. It is a gridworld with sparse reward at a goal state. We compare naive epsilon-greedy exploration to an optimistic UCB-style bonus. The demo is meant to show how optimism tends to push the agent toward informative regions faster than random exploration.

## Slide 18: Demo Output
The output shows two things: cumulative regret over episodes and heatmaps of state visits. The optimistic agent usually reaches goal-directed regions faster, so its cumulative regret is lower, and its state-visit heatmap shows more focused coverage of promising paths.

## Slide 19: Key Takeaways
There are three takeaways to emphasize. First, this is the first provably efficient RL framework for nonlinear function approximation. Second, the optimism principle scales beyond tabular and linear settings into kernels and wide neural nets. Third, NTK theory provides a bridge between deep networks and kernel-style guarantees, narrowing the theory-practice gap.

## Slide 20: Limitations and Open Questions
The assumptions are also important. Bellman completeness is quite restrictive in practice. Kernel eigenvalue decay is assumed known. And the neural network guarantees require very wide networks, which is far from standard deep learning. Computationally, the kernel method requires Gram matrix inversion, which scales cubically in the data size. Open directions include relaxing the structural assumptions, making the neural theory hold at practical widths, and reducing computational cost.

## Slide 21: References and Closing
I will close with the key references: the main Yang et al. paper, the linear approximation results by Jin et al., the neural tangent kernel paper by Jacot and coauthors, and the classic UCB and GP-UCB bandit papers. Thank you, and I am happy to take questions.

