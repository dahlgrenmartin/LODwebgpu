import assert from 'node:assert/strict';
import { findResolution, formatSupportedSizes } from '../src/manifest.ts';

const manifest = {
  resolutions: [
    { width: 256, height: 256, latent: [1, 32, 32, 32], numel: 32768, graph: 'lod_joint_256x256.onnx' },
    { width: 1024, height: 768, latent: [1, 32, 96, 128], numel: 393216, graph: 'lod_joint_1024x768.onnx' },
    { width: 768, height: 1024, latent: [1, 32, 128, 96], numel: 393216, graph: 'lod_joint_768x1024.onnx' },
  ],
  weights: 'weights.bin', encoder: 'encoder.onnx', encoderWeights: 'encoder_weights.bin',
  adam: { lr: 0.03, beta1: 0.9, beta2: 0.999, eps: 1e-8, steps: 30 },
  inputs: ['z', 'target'], outputs: ['loss', 'pred', 'score', 'grad_z'],
};

assert.equal(findResolution(manifest, 1024, 768)?.graph, 'lod_joint_1024x768.onnx');
assert.equal(findResolution(manifest, 768, 1024)?.graph, 'lod_joint_768x1024.onnx');
assert.equal(findResolution(manifest, 512, 512), undefined);
assert.equal(
  formatSupportedSizes(manifest.resolutions),
  '256x256, 1024x768, 768x1024',
);

console.log('ORT shape selection tests passed');
