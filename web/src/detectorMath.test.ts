import { bandPsnr, linearSlope } from './detectorMath';

function close(got: number, want: number, eps = 1e-12): void {
  if (Math.abs(got - want) > eps) {
    throw new Error(`expected ${want}, got ${got}`);
  }
}

close(bandPsnr(10), -10);
close(linearSlope([1, 2, 3, 4], 10), 1);
close(linearSlope([9, 7, 5, 3, 1], 3), -2);
close(linearSlope([3, 3, 3, 3], 4), 0);
if (!Number.isNaN(linearSlope([1], 10))) {
  throw new Error('a slope needs at least two values');
}

console.log('detectorMath tests passed');
