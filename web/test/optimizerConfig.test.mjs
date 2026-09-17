import assert from 'node:assert/strict';
import { parseOptimizerConfig } from '../src/optimizerConfig.ts';

const good = parseOptimizerConfig({
  lr: '0.03', beta1: '0.9', beta2: '0.999', eps: '1e-8', steps: '30',
});
assert.deepEqual(good, { lr: 0.03, beta1: 0.9, beta2: 0.999, eps: 1e-8, steps: 30 });

for (const [field, value] of [
  ['lr', '0'], ['lr', 'NaN'], ['beta1', '1'], ['beta1', '-0.1'],
  ['beta2', '1'], ['eps', '0'], ['steps', '1'], ['steps', '2.5'],
]) {
  const input = {
    lr: '0.03', beta1: '0.9', beta2: '0.999', eps: '1e-8', steps: '30',
    [field]: value,
  };
  assert.throws(() => parseOptimizerConfig(input), new RegExp(field, 'i'));
}

console.log('optimizerConfig tests passed');
