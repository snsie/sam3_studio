/** Small seeded PRNG so synthetic lecture data looks identical on every reload. */
export type Random = () => number;

/** mulberry32: fast 32-bit generator, plenty for synthetic images. */
export function createRandom(seed: number): Random {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomBetween(random: Random, min: number, max: number) {
  return min + (max - min) * random();
}

/** Standard normal sample (Box-Muller). */
export function randomNormal(random: Random) {
  const u = Math.max(random(), 1e-12);
  const v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
