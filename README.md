# GroupedNewton

Code associated to the paper *Adapting Newton's Method to Neural Networks through a Summary of Higher-Order Derivatives* (2023), P. Wolinski.

Link: [https://arxiv.org/abs/2312.03885](https://arxiv.org/abs/2312.03885).

# Quick reminder

We consider a loss $\mathcal{L}(\boldsymbol{\theta})$ to minimize according to
a vector of parameters $\boldsymbol{\theta} \in \mathbb{R}^P$. We represent this vector by
a tuple of $S$ subsets (or groups) of parameters $(\mathbf{T}_1, \cdots, \mathbf{T}_S)$. This tuple can be seen
as a partition of the set of the indices $\{1, \cdots, P\}$ of the vector $\boldsymbol{\theta}$,
so that each parameter $\boldsymbol{\theta}_p$ belongs to exactly one subset (or group) $\mathbf{T}_s$.
We assume that $S \ll P$. 

For a direction of descent $\mathbf{u}_t$, we consider a training step where a learning rate
$\eta_s$ is assigned to the update of the parameters belonging to $\mathbf{T}_s$.
By compiling the $\eta_s$ into one vector $\boldsymbol{\eta} = (\eta_1, \cdots, \eta_S) \in \mathbb{R}^S$, we have the following training step:
```math
\boldsymbol{\theta}_{t + 1} = \boldsymbol{\theta}_t - \mathbf{U}_t \mathbf{I}_{P:S} \boldsymbol{\eta}_t ,
```
where $`\mathbf{U}_t = \mathrm{Diag}(\mathbf{u}_t)`$, $`\mathbf{I}_{S:P}`$ is the partition matrix: $`(\mathbf{I}_{S:P})_{sp} = 1`$ iff $`\theta_p \in \mathbf{T}_s`$ else $0$, 
$`\mathbf{I}_{P:S} = \mathbf{I}_{S:P}^T`$.

**Our goal is to build a vector $\boldsymbol{\eta}$ of per-subset learning rates $\eta_s$,
optimizing the decrease of the loss at each training step. For that, 
we use order-2 and order-3 information on the loss.**

The second-order Taylor approximation of $\mathcal{L}(\boldsymbol{\theta}_{t+1})$ around $`\boldsymbol{\theta}_{t}`$ gives:
```math
\begin{aligned}
\mathcal{L}(\boldsymbol{\theta}_{t+1}) - \mathcal{L}(\boldsymbol{\theta}_{t}) \approx
\boldsymbol{\Delta}_2(\boldsymbol{\eta}_t) &:= 
-\mathbf{g}_t^T \mathbf{U}_t \mathbf{I}_{P:S} \boldsymbol{\eta}_t
	+ \frac{1}{2} \boldsymbol{\eta}_t^T \mathbf{I}_{S:P} \mathbf{U}_t \mathbf{H}_t \mathbf{U}_t \mathbf{I}_{P:S} \boldsymbol{\eta}_t \\
&= \bar{\mathbf{g}}^T \boldsymbol{\eta}_t + \frac{1}{2} \boldsymbol{\eta}_t^T \bar{\mathbf{H}} \boldsymbol{\eta}_t ,
\end{aligned}
```
where:
```math
\bar{\mathbf{H}}_t = \mathbf{I}_{S:P} \mathbf{U}_t \mathbf{H}_t \mathbf{U}_t \mathbf{I}_{P:S} , \qquad
\bar{\mathbf{g}}_t = \mathbf{I}_{S:P} \mathbf{U}_t \mathbf{g}_t .
```
Therefore, the minimum of the second-order approximation $\boldsymbol{\Delta}_2$ of the loss $\mathcal{L}$ is
attained for $\boldsymbol{\eta} = \bar{\mathbf{H}}_t^{-1} \bar{\mathbf{g}}_t$.

Besides, in our method, we regularize $\bar{\mathbf{H}}$ by using order-3 information on the loss, 
which leads use to the *anisotropic Nesterov cubic regularization* scheme. Finally, 
the training step of our method is: $`\boldsymbol{\theta}_{t + 1} = \boldsymbol{\theta}_t - \mathbf{U}_t \mathbf{I}_{P:S} \boldsymbol{\eta}_t^*`$, where
$`\boldsymbol{\eta}_t^*`$ is the solution of largest norm $||\mathbf{D}_t \boldsymbol{\eta}||$ of the equation:
```math
\boldsymbol{\eta} = \left(\bar{\mathbf{H}}_t + \frac{\lambda_{\mathrm{int}}}{2} \|\mathbf{D}_t \boldsymbol{\eta}\| \mathbf{D}_t^2\right)^{-1}\bar{\mathbf{g}}_t ,
```
where $\lambda_{\mathrm{int}}$ is the internal damping, which is a hyperparameter to set, and:
```math
\mathbf{D}_t = \mathrm{Diag}\left(|\mathbf{D}^{(3)}_{\boldsymbol{\theta}_t}(\mathbf{u}_t)|^{1/3}_{iii} : i \in \{1, \cdots, S\}\right).
```
See Section 3 for a formal defintion of $\mathbf{D}^{(3)}_{\boldsymbol{\theta}_t}(\mathbf{u}_t)$.

