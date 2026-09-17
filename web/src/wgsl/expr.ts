/**
 * Symbolic integer expressions over the graph's shape symbols.
 *
 * The exported graph never bakes an image size: any integer that depends on the
 * input dimensions arrives as one of these trees. Binding the symbols from the
 * actual input tensor and evaluating them is what makes the runtime accept any
 * image size, which is the whole point of this backend.
 */

export type Expr =
  | number
  | { sym: string }
  | { op: 'add' | 'mul' | 'pow' | 'floordiv' | 'mod'; args: Expr[] };

export type Bindings = Record<string, number>;

export function evalExpr(e: Expr, b: Bindings): number {
  if (typeof e === 'number') return e;
  if ('sym' in e) {
    const v = b[e.sym];
    if (v === undefined) throw new Error(`unbound shape symbol ${e.sym}`);
    return v;
  }
  const a = e.args.map((x) => evalExpr(x, b));
  switch (e.op) {
    case 'add': return a.reduce((x, y) => x + y, 0);
    case 'mul': return a.reduce((x, y) => x * y, 1);
    case 'pow': return Math.pow(a[0], a[1]);
    // Python floor semantics: -7 // 2 === -4, not -3.
    case 'floordiv': return Math.floor(a[0] / a[1]);
    case 'mod': return ((a[0] % a[1]) + a[1]) % a[1];
    default: throw new Error(`unknown expression op ${(e as any).op}`);
  }
}

export function evalShape(shape: Expr[], b: Bindings): number[] {
  return shape.map((d) => evalExpr(d, b));
}

/**
 * Solve the symbol bindings from a concrete input shape.
 *
 * Only the simple forms the exporter actually emits are handled: a bare symbol,
 * or a symbol scaled by a constant (the target is 8x the latent). Anything else
 * is refused rather than guessed at.
 */
export function bindFromShape(
  declared: Expr[], actual: number[], into: Bindings,
): Bindings {
  if (declared.length !== actual.length) {
    throw new Error(`rank mismatch: declared ${declared.length}, got ${actual.length}`);
  }
  for (let i = 0; i < declared.length; i++) {
    const d = declared[i];
    const v = actual[i];
    if (typeof d === 'number') {
      if (d !== v) throw new Error(`dim ${i} must be ${d}, got ${v}`);
      continue;
    }
    if ('sym' in d) {
      assign(into, d.sym, v, i);
      continue;
    }
    if (d.op === 'mul' && d.args.length === 2) {
      const [x, y] = d.args;
      const konst = typeof x === 'number' ? x : typeof y === 'number' ? y : null;
      const symNode = typeof x === 'number' ? y : x;
      if (konst !== null && typeof symNode !== 'number' && 'sym' in symNode) {
        if (v % konst !== 0) {
          throw new Error(
            `dim ${i} is ${v}, which is not a multiple of ${konst}; ` +
            `the decoder upsamples by exactly ${konst}`);
        }
        assign(into, symNode.sym, v / konst, i);
        continue;
      }
    }
    throw new Error(`cannot solve symbol from dim ${i} of shape`);
  }
  return into;
}

function assign(into: Bindings, sym: string, value: number, dim: number): void {
  if (value <= 0 || !Number.isInteger(value)) {
    throw new Error(`dim ${dim} resolved symbol ${sym} to ${value}`);
  }
  const prev = into[sym];
  if (prev !== undefined && prev !== value) {
    throw new Error(`symbol ${sym} bound to ${prev} and ${value}`);
  }
  into[sym] = value;
}
