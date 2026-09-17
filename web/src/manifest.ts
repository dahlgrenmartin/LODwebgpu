export interface AdamConfig {
  lr: number; beta1: number; beta2: number; eps: number; steps: number;
}

export interface Resolution {
  image: number; latent: number[]; numel: number; graph: string;
}

export interface Manifest {
  resolutions: Resolution[]; weights: string; encoder: string;
  adam: AdamConfig; inputs: string[]; outputs: string[];
}

const ADAM_KEYS: (keyof AdamConfig)[] = ['lr', 'beta1', 'beta2', 'eps', 'steps'];

/**
 * Load and validate the manifest.
 *
 * The Adam hyperparameters have no defaults on purpose: they must match the
 * reference LOD implementation, and a silently-defaulted value would produce a
 * plausible-looking run that is not comparable to anything.
 */
export async function loadManifest(url: string): Promise<Manifest> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`manifest fetch failed: ${res.status}`);
  const m = (await res.json()) as Manifest;
  if (!m.adam) throw new Error('manifest.adam missing');
  for (const k of ADAM_KEYS) {
    if (typeof m.adam[k] !== 'number') {
      throw new Error(`manifest.adam.${k} is required and has no default`);
    }
  }
  if (!m.resolutions?.length) throw new Error('manifest.resolutions empty');
  return m;
}
