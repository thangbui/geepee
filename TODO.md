## General

- [ ] save/load models: is saving parameters enough?
- [x] remove for loops in aep_models update_posterior, compute_cavity
- [ ] separate parameters for different output dimensions

## AEP (BB-alpha) modules

- [x] sparse GPLVM
- [x] sparse GP regression and classification: make sure that the gradients of logZ 
are the same with sparse GPLVM when variance = 0
- [x] sparse Deep GP regression and classification
- [ ] sparse Deep GP latent variable models
- [ ] for sparse Deep GP: alpha = 1.0 do moment matching, 
for other alphas, what do we do?
- [ ] moment matching vs linearisation vs quadrature/MC

## EP modules

- [x] sparse GPR/C inference
- [ ] sparse GPR/C learning
- [ ] GPLVM inference
- [ ] GPLVM learning
- [ ] Deep-GP inference
- [ ] Deep-GP learning
- [ ] GP SSM inference
- [ ] GP SSM learning
- [ ] should work with all alphas

## VI modules

- [x] Titsias's VI for GPR/C
- [ ] Titsias's VI for GPLVM (collapsed bound)
- [ ] Frigola's VI for GPSSM
- [ ] Hensman's VI compression for Deep GPR/C, LVM
- [ ] make sure that these converge to AEP/EP modules when alpha -> 0