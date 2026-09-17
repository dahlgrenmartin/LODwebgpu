export interface AdamConfig {
  lr: number; beta1: number; beta2: number; eps: number; steps: number;
}

export interface Resolution {
  width: number; height: number; latent: number[]; numel: number; graph: string;
}

export interface Manifest {
  resolutions: Resolution[]; weights: string; encoder: string;
  encoderWeights: string;
  adam: AdamConfig; inputs: string[]; outputs: string[];
}

const ADAM_KEYS: (keyof AdamConfig)[] = ['lr', 'beta1', 'beta2', 'eps', 'steps'];

export function findResolution(
  manifest: Pick<Manifest, 'resolutions'>, width: number, height: number,
): Resolution | undefined {
  return manifest.resolutions.find((r) => r.width === width && r.height === height);
}

export function formatSupportedSizes(resolutions: readonly Resolution[]): string {
  return resolutions.map((r) => `${r.width}x${r.height}`).join(', ');
}

/**
 * Load and validate the manifest.
 *
 * The Adam hyperparameters have no defaults on purpose: they must match the
 * reference LOD implementation unless the user explicitly overrides them in the
 * browser controls.
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
  for (const r of m.resolutions) {
    if (!Number.isInteger(r.width) || r.width <= 0 ||
        !Number.isInteger(r.height) || r.height <= 0) {
      throw new Error('manifest resolution width/height must be positive integers');
    }
    if (!Array.isArray(r.latent) || r.latent.length !== 4 || typeof r.graph !== 'string') {
      throw new Error(`invalid manifest resolution ${r.width}x${r.height}`);
    }
  }
  return m;
}
