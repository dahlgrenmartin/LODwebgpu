import type { AdamConfig } from './manifest';

export type OptimizerConfigText = Record<keyof AdamConfig, string>;

function finite(name: keyof AdamConfig, raw: string): number {
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`${name} must be a finite number`);
  return value;
}

export function parseOptimizerConfig(input: OptimizerConfigText): AdamConfig {
  const lr = finite('lr', input.lr);
  const beta1 = finite('beta1', input.beta1);
  const beta2 = finite('beta2', input.beta2);
  const eps = finite('eps', input.eps);
  const steps = finite('steps', input.steps);

  if (lr <= 0) throw new Error('lr must be > 0');
  if (beta1 < 0 || beta1 >= 1) throw new Error('beta1 must be in [0, 1)');
  if (beta2 < 0 || beta2 >= 1) throw new Error('beta2 must be in [0, 1)');
  if (eps <= 0) throw new Error('eps must be > 0');
  if (!Number.isInteger(steps) || steps < 2) {
    throw new Error('steps must be an integer >= 2');
  }

  return { lr, beta1, beta2, eps, steps };
}
