# Related paper

Code associated to the paper *Adapting Newton's Method to Neural Networks through a Summary of Higher-Order Derivatives* (2023), P. Wolinski.

Link: [https://arxiv.org/abs/2312.03885](https://arxiv.org/abs/2312.03885).

# Tutorial

## Minimal example

```python
import torch
import grnewt

# User-specific
data_loader = torch.utils.data.DataLoader(...)
model = MyModel(...)
loss_fn = lambda output, target: ...

# Create a specific data loader
hg_loader = torch.utils.data.DataLoader(...)

# Set some hyperparameters
damping = .1
damping_int = 10.

# Prepare the optimizer
full_loss = lambda x, target: loss_fn(model(x), target)
param_groups, name_groups = grnewt.partition.canonical(model)
optimizer = grnewt.NewtonSummary(param_groups, full_loss, hg_loader, 
                    damping = damping, dct_nesterov = {'use': True, 'damping_int': damping_int}, 
                    period_hg = 10, mom_lrs = .5, remove_negative = True,
                    momentum = .9, momentum_damp = .9)

# Optimization process
for epoch in range(10):
    for x, target in data_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()
```

## Explanations

Specific variables:
 * `param_groups`: partition of the set of parameters; given parameters `t1`, `t2`, ..., one may set:
   * `param_groups = [{'params': [t1, t2]}, {'params': [t3]}, {'params': [t4, t5, t6]}]`,
   * or simply use a predefined partition: `param_groups, name_groups = grnewt.partition.canonical(model)`;
 * `data_loader`: loader used to compute the gradient of the loss and use it to propose a descent direction $\mathbf{u}$;
 * `hg_loader`: loader used to compute $\bar{\mathbf{H}}$, $\bar{\mathbf{g}}$ and $\bar{\mathbf{D}}$ in direction $\mathbf{u}$, in order to obtain the vector of learning rates $\boldsymbol{\eta}$;
 * `period_hg`: period of updates of $\bar{\mathbf{H}}$, $\bar{\mathbf{g}}$, $\bar{\mathbf{D}}$ and $\boldsymbol{\eta}^*$;
 * `damping`: damping, or global learning rates factor $\lambda_1$;
 * `damping_int`: internal damping $\lambda_{\mathrm{int}}$, strength of regularization of $\bar{\mathbf{H}}$ when using anisotropic Nesterov regularization;
 * `mom_lrs`: factor of the moving average used to update $\boldsymbol{\eta}^*$, in order to smooth the trajectory of the computed learning rates;
 * `remove_negative`: set negative learning rates to zero.

# Quick reminder

We consider a loss $\mathcal{L}(\boldsymbol{\theta})$ to minimize according to
a vector of parameters $\boldsymbol{\theta} \in \mathbb{R}^P$. We represent this vector by
a tuple of $S$ subsets of parameters $(\mathbf{T}_1, \cdots, \mathbf{T}_S)$. This tuple can be seen
as a partition of the set of the indices $\{1, \cdots, P\}$ of the vector $\boldsymbol{\theta}$,
so that each parameter $\boldsymbol{\theta}_p$ belongs to exactly one subset $\mathbf{T}_s$.
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
See Section 3 of the paper for a formal defintion of $\mathbf{D}^{(3)}_{\boldsymbol{\theta}_t}(\mathbf{u}_t)$.

# Using the code

## Structure

Package `grnewt`:
 * `partition`: tools for building a partition of the set of parameters:
   * `canonical`: creates a per-tensor partition,
   * `trivial`: creates a partition with only one subset, containing all the parameters,
   * `wb`: creates a partition with 3 subsets: subsets of weights, subset of biases, subset of all remaining parameters;
 * `hg.compute_Hg`: computes $\bar{\mathbf{H}}$, $\bar{\mathbf{g}}$ and $\mathbf{D}$;
 * `nesterov.nesterov_lrs`: computes $\boldsymbol{\eta}^*$ with the anisotropic Nesterov regularization scheme, by using $\bar{\mathbf{H}}$, $\bar{\mathbf{g}}$ and $\mathbf{D}$;
 * `newton_summary.NewtonSummary`: class containing the main optimizer, implementing the optimization procedure described in the paper (see Appendix F);
 * `util.fullbatch.fullbatch_gradient`: computes the fullbatch gradient of the loss, given a model, a loss and a dataset; useful for proposing a direction of descent $\mathbf{u}$;
 * `models`: sub-package containing usual models;
 * `datasets`: sub-package containing usual datasets and dataset tools.
