## General

- [ ] save/load models: is saving parameters enough?
- [x] remove for loops in aep_models update_posterior, compute_cavity
- [ ] separate parameters for different output dimensions
- [ ] stochastic optimisations for all models, 
check that gradients are correct and bias is small
- [ ] moment matching vs linearisation vs quadrature/MC

## AEP (BB-alpha) modules

- [x] sparse GPLVM
- [x] sparse GP regression and classification: make sure that the gradients of logZ 
are the same with sparse GPLVM when variance = 0
- [x] sparse Deep GP regression and classification
- [ ] sparse Deep GP latent variable models
- [ ] sparse Deep GP with hidden variables
- [x] sparse GP state space models
- [ ] for sparse Deep GP with no hidden variables: alpha = 1.0 do moment matching, 
for other alphas, what do we do?

## EP modules
- [ ] numerical instability/invalid updates
- [x] sparse GPR/C inference
- [ ] sparse GPR/C learning
- [x] GPLVM inference
- [ ] GPLVM learning, half-ticked, could use AEP learning insteard
- [ ] Deep-GP inference
- [ ] Deep-GP learning
- [x] GP SSM inference
- [ ] GP SSM learning, half-ticked, could use AEP learning instead
- [ ] GP SSM inference and learning with control signal
- [ ] should work with all alphas

## VI modules

- [x] Titsias's VI for GPR/C
- [ ] Titsias's VI for GPLVM (collapsed bound)
- [ ] Frigola's VI for GPSSM
- [ ] Hensman's VI compression for Deep GPR/C, LVM
- [ ] make sure that these converge to AEP/EP modules when alpha -> 0
