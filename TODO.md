## General

- [ ] save/load models: is saving parameters enough?

## AEP modules

- [x] sparse GPLVM
- [x] sparse GP regression and classification: make sure that the gradients of logZ 
are the same with sparse GPLVM when variance = 0
- [ ] sparse Deep GP regression and classification
- [ ] sparse Deep GP latent variable models
- [ ] for sparse Deep GP: alpha = 1.0 do moment matching, 
for other alphas, what do we do?
- [ ] moment matching vs linearisation vs quadrature

## EP modules

- [ ] separate inference and learning
- [ ] sparse GPR/C, GPLVM, Deep-GP, GP SSM
- [ ] should work with all alphas

## VI modules

- [ ] Titsias's VI for GPR/C
- [ ] Titsias's VI for GPLVM (collapsed bound)
- [ ] Frigola's VI for GPSSM
- [ ] Hensman's VI compression for Deep GPR/C, LVM
- [ ] make sure that these converge to AEP/EP modules when alpha -> 0