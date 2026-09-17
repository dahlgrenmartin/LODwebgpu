# Milestone 1 convergence record

**Date:** 2026-09-17
**Adapter:** nvidia / blackwell (RTX 5090), Chrome 152, ORT-Web 1.30.0 WebGPU EP
**Resolution:** 128x128 (latent [1,32,16,16], 8192 elements)

> **These use FIXTURE hyperparameters, not the reference LOD values.**
> `lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, steps=30`. The numbers below show the
> loop is mechanically correct; they are **not** a statement about LOD's behaviour.
> The real values must come from the reference implementation before any claim is
> made about convergence quality or detector scores.

## Result

Loss decreases monotonically across all 30 steps:

```
0.742496  0.735125  0.729355  0.724831  0.721240  0.718411
0.716025  0.713920  0.711992  0.710191  0.708457  0.706828
0.705287  0.703827  0.702474  0.701195  0.699952  0.698762
0.697625  0.696535  0.695442  0.694342  0.693296  0.692284
0.691295  0.690335  0.689401  0.688500  0.687619  0.686753
```

Total improvement 0.0557 (7.5%). Detector score 0.259143 -> 0.259433.

The small improvement and near-static score are what fixture hyperparameters on a
random-weight decoder predict; neither is evidence about LOD.

## Performance

**52.8 ms/step**, so ~1.6 s for the full 30 steps at 128x128 on an RTX 5090.

This is the headless loop, which reads `z` back to the CPU every step. Milestone 2
replaces that with a gpu-buffer input binding, which should remove a full
round-trip per step.

Extrapolating to the demo's targets: 256x256 is 4x the decoder work, and an
integrated GPU is commonly 10-20x slower than a 5090. That puts a naive 256x256
run on integrated graphics in the tens of seconds. The resolution-selection and
downgrade path designed in spec section 5 is therefore load-bearing, not optional,
and should be measured on real integrated hardware early in Milestone 2.
