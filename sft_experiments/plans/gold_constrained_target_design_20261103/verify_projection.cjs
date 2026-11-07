/* CPU-only mathematical verification; no model inference or training. */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');

let rngState = 20261103;
function rng() {
  rngState = (Math.imul(1664525, rngState) + 1013904223) >>> 0;
  return (rngState + 0.5) / 4294967296;
}
function sum(a) { return a.reduce((v, x) => v + x, 0); }
function softmax(z) {
  let m = -Infinity;
  for (const v of z) m = Math.max(m, v);
  const e = z.map(v => Math.exp(v - m));
  const d = sum(e);
  return e.map(v => v / d);
}
function kl(p, q) { return sum(p.map((v, i) => v * Math.log(v / q[i]))); }
function maxDiff(a, b) { return Math.max(...a.map((v, i) => Math.abs(v - b[i]))); }
function argmax(a) { return a.reduce((b, v, i) => v > a[b] ? i : b, 0); }
function target(p, w) { return softmax(p.map((v, i) => Math.log(v) + w[i])); }

// F(tau) is strictly increasing. Taking the upper bracket is conservative.
function solveBisection(p, s, g, iterations = 48) {
  let lo = s[g];
  let hi = lo;
  for (const v of s) hi = Math.max(hi, v);
  for (let k = 0; k < iterations; k++) {
    const mid = (lo + hi) / 2;
    let f = p[g] * (mid - s[g]);
    for (let i = 0; i < p.length; i++) {
      if (i !== g) f -= p[i] * Math.max(s[i] - mid, 0);
    }
    if (f < 0) lo = mid; else hi = mid;
  }
  const tau = hi;
  const w = s.map((v, i) => i === g ? tau : Math.min(v, tau));
  return { tau, w, q: target(p, w) };
}

// Independent active-set solution: pool high-score candidates with gold.
function solveSorted(p, s, g) {
  const order = s.map((_, i) => i).filter(i => i !== g)
    .sort((a, b) => s[b] - s[a]);
  let weight = p[g];
  let numerator = p[g] * s[g];
  let tau = s[g];
  for (const i of order) {
    if (s[i] <= tau) break;
    weight += p[i];
    numerator += p[i] * s[i];
    tau = numerator / weight;
  }
  const w = s.map((v, i) => i === g ? tau : Math.min(v, tau));
  return { tau, w, q: target(p, w) };
}

const metrics = {
  cases: 0, goldTop1Cases: 0,
  maxNormalizationError: 0, maxGoldProbabilityDecrease: 0,
  maxLogOddsViolation: 0, maxSolverDisagreement: 0,
  maxWeightedKktResidual: 0, maxObjectiveExcessOverTeacher: 0,
  maxObjectiveExcessOverFeasibleCandidate: 0,
  maxUnchangedFeasibleTargetError: 0,
};
function check(p, s, g, compareSorted = true) {
  const { tau, w, q } = solveBisection(p, s, g);
  const raw = target(p, s);
  assert(q.every(v => Number.isFinite(v) && v > 0));
  const normalization = Math.abs(sum(q) - 1);
  const goldDecrease = Math.max(p[g] - q[g], 0);
  let oddsViolation = 0;
  let weightedKkt = p[g] * (tau - s[g]);
  for (let i = 0; i < p.length; i++) {
    if (i === g) continue;
    oddsViolation = Math.max(oddsViolation,
      Math.log(q[i]) - Math.log(q[g]) - Math.log(p[i]) + Math.log(p[g]));
    const multiplier = Math.max(s[i] - tau, 0);
    assert(Math.abs(w[i] - s[i] + multiplier) < 1e-12);
    if (multiplier > 0) assert(Math.abs(w[i] - tau) < 1e-12);
    weightedKkt -= p[i] * multiplier;
  }
  const objective = kl(q, raw);
  const excess = objective - kl(p, raw);
  let diff = 0;
  if (compareSorted) {
    const independent = solveSorted(p, s, g);
    diff = Math.max(Math.abs(tau - independent.tau), maxDiff(q, independent.q));
    assert(diff < 1e-11);
  }
  // A feasible one-sided clipping candidate, plus random feasible scores.
  const alternatives = [s.map(v => Math.min(v, s[g]))];
  if (p.length < 1000) {
    for (let k = 0; k < 3; k++) alternatives.push(p.map((_, i) => i === g ? 0 : -3 * rng()));
  }
  let alternativeExcess = 0;
  for (const a of alternatives) {
    alternativeExcess = Math.max(alternativeExcess, objective - kl(target(p, a), raw));
  }
  assert(normalization < 1e-10);
  assert(goldDecrease < 1e-12);
  assert(oddsViolation < 1e-10);
  assert(Math.abs(weightedKkt) < 1e-11);
  assert(excess < 1e-10);
  assert(alternativeExcess < 1e-10);
  if (argmax(p) === g) {
    assert(argmax(q) === g);
    metrics.goldTop1Cases++;
  }
  let feasibleError = 0;
  if (s.every(v => v <= s[g])) {
    feasibleError = maxDiff(q, raw);
    assert(feasibleError < 1e-12);
  }
  metrics.cases++;
  metrics.maxNormalizationError = Math.max(metrics.maxNormalizationError, normalization);
  metrics.maxGoldProbabilityDecrease = Math.max(metrics.maxGoldProbabilityDecrease, goldDecrease);
  metrics.maxLogOddsViolation = Math.max(metrics.maxLogOddsViolation, oddsViolation);
  metrics.maxSolverDisagreement = Math.max(metrics.maxSolverDisagreement, diff);
  metrics.maxWeightedKktResidual = Math.max(metrics.maxWeightedKktResidual, Math.abs(weightedKkt));
  metrics.maxObjectiveExcessOverTeacher = Math.max(metrics.maxObjectiveExcessOverTeacher, excess);
  metrics.maxObjectiveExcessOverFeasibleCandidate = Math.max(metrics.maxObjectiveExcessOverFeasibleCandidate, alternativeExcess);
  metrics.maxUnchangedFeasibleTargetError = Math.max(metrics.maxUnchangedFeasibleTargetError, feasibleError);
  return { p, s, g, tau, raw, projected: q, objective };
}

const toyP = [0.6, 0.3, 0.1];
const toyR = [0.65, 0.01, 0.34];
const toyScore = toyP.map((v, i) => 0.5 * Math.max(-2, Math.min(2, Math.log(v / toyR[i]))));
const toy = { ...check(toyP, toyScore, 0), r: toyR, beta: 0.5, clip: 2 };
const floorOnlyCounterexample = {
  p: [0.4, 0.35, 0.25], raw: [0.2, 0.7, 0.1],
  goldFloorOnlyProjection: [0.4, 0.525, 0.075], gold: 0,
};
assert(argmax(floorOnlyCounterexample.p) === 0);
assert(argmax(floorOnlyCounterexample.goldFloorOnlyProjection) !== 0);

check([0.2, 0.7, 0.1], [0, 0, 0], 0);
check([0.2, 0.7, 0.1], [1, -0.2, 0.2], 0);
const binary = check([0.6, 0.4], [-1, 1], 0);
assert(maxDiff(binary.p, binary.projected) < 1e-12);
check([1e-25, 0.25, 0.75 - 1e-25], [-1, 1, 0], 0);

for (let k = 0; k < 2000; k++) {
  const n = 3 + Math.floor(126 * rng());
  const p = softmax(Array.from({ length: n }, () => 20 * (rng() - 0.5)));
  const s = Array.from({ length: n }, () => 2 * rng() - 1);
  const g = k % 4 === 0 ? argmax(p) : Math.floor(n * rng());
  check(p, s, g);
}

// Full-vocabulary-size synthetic numerical case; not a GPU timing benchmark.
const fullP = softmax(Array.from({ length: 161280 }, () => 20 * (rng() - 0.5)));
const fullS = Array.from({ length: 161280 }, () => 2 * rng() - 1);
check(fullP, fullS, argmax(fullP), false);

// Forward-KL logit gradient, checked independently with finite differences.
const logits = [0.7, -0.2, -0.8];
const student = softmax(logits);
const q = toy.projected;
const analytic = student.map((v, i) => v - q[i]);
const eps = 1e-5;
const numerical = logits.map((_, i) => {
  const plus = [...logits], minus = [...logits];
  plus[i] += eps; minus[i] -= eps;
  return (kl(q, softmax(plus)) - kl(q, softmax(minus))) / (2 * eps);
});
const gradientError = maxDiff(analytic, numerical);
assert(gradientError < 1e-9);
const hardCe = student.map((v, i) => v - (i === 0 ? 1 : 0));
const scalar = sum(analytic.map((v, i) => v * hardCe[i])) / sum(hardCe.map(v => v * v));
const nonCollinearResidual = Math.sqrt(sum(analytic.map((v, i) => (v - scalar * hardCe[i]) ** 2)));
assert(nonCollinearResidual > 0.01);

const matchGold = toy.p.map((v, i) => i === 0 ? q[0] : (1 - q[0]) * v / (1 - toy.p[0]));
const report = {
  createdAt: new Date().toISOString(), status: 'passed', seed: 20261103,
  scope: 'CPU synthetic mathematical verification only. No model forward passes, no training, no revised diagnostic accuracy.',
  sourceSha256: crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex'),
  floatType: 'JavaScript IEEE-754 binary64', bisectionIterations: 48,
  metrics, toy, floorOnlyCounterexample,
  confidenceMatchedControl: { distribution: matchGold, klFromProjectedTarget: kl(q, matchGold) },
  gradient: { student, analytic, numerical, maxError: gradientError,
    hardCe, bestScalar: scalar, nonCollinearResidual },
  limitations: [
    'Constraints preserve gold odds relative to the ordinary has-reasoning teacher; they do not force gold to rank first if that teacher was wrong.',
    'Gold-NLL nondegradation is a construction guarantee, not evidence of improved accuracy.',
    'The target can equal the teacher; non-gold reasoning signal retention must be measured on real data.',
    'Finite KL anchoring does not guarantee preservation of no-reasoning accuracy under shared parameters.',
    'CPU double precision verification is not verification of a future GPU FP32 implementation.',
  ],
};
fs.writeFileSync(path.join(__dirname, 'numerical_verification.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ status: report.status, metrics, toy, gradientError, nonCollinearResidual }, null, 2));
