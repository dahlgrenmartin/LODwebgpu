export function linearSlope(values: readonly number[], window: number): number {
  const y = values.slice(-window);
  const n = y.length;
  if (n < 2) return Number.NaN;

  // Same ordinary least-squares fit as sklearn LinearRegression over x=0..n-1.
  const xMean = (n - 1) / 2;
  let yMean = 0;
  for (const v of y) yMean += v;
  yMean /= n;

  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < n; i++) {
    const dx = i - xMean;
    numerator += dx * (y[i] - yMean);
    denominator += dx * dx;
  }
  return numerator / denominator;
}
