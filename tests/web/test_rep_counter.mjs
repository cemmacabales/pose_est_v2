import assert from 'assert/strict';
import { RepCounter } from '../../web/rep_counter.js';

// EXERCISE_REP_CONFIG for "Deep Squat":
//   features: [2, 3] (L_knee, R_knee cosines)
//   direction: "low", enter: 0.5, exit: 0.8
// EMA alpha = 0.3. Rep counted when smoothed goes below 0.5 then comes back above 0.8.

function makeAngles(kneeVal) {
  // 16-feature angles array; only indices 2 and 3 (knee cosines) matter for Deep Squat
  const a = new Float32Array(16).fill(0);
  a[2] = kneeVal;
  a[3] = kneeVal;
  return a;
}

// Feed enough frames of straight knees (0.9) → bent knees (0.2) → straight again (0.9)
// to trigger one rep count.
{
  const rc = new RepCounter();
  // 5 frames straight
  for (let i = 0; i < 5; i++) rc.update('Deep Squat', makeAngles(0.9));
  assert.equal(rc.get('Deep Squat'), 0, 'no rep yet — still straight');

  // 10 frames bent (drives smoothed below 0.5)
  for (let i = 0; i < 10; i++) rc.update('Deep Squat', makeAngles(0.2));
  assert.equal(rc.get('Deep Squat'), 0, 'no rep yet — peaked but not returned');

  // 15 frames straight (drives smoothed above 0.8)
  for (let i = 0; i < 15; i++) rc.update('Deep Squat', makeAngles(0.9));
  assert.equal(rc.get('Deep Squat'), 1, 'one rep counted after return to straight');

  console.log('Deep Squat rep counting: PASS');
}

// Second rep in the same session
{
  const rc = new RepCounter();
  for (let rep = 0; rep < 2; rep++) {
    for (let i = 0; i < 5; i++)  rc.update('Deep Squat', makeAngles(0.9));
    for (let i = 0; i < 10; i++) rc.update('Deep Squat', makeAngles(0.2));
    for (let i = 0; i < 15; i++) rc.update('Deep Squat', makeAngles(0.9));
  }
  assert.equal(rc.get('Deep Squat'), 2, 'two reps counted');
  console.log('Two reps: PASS');
}

// Unknown exercise → get returns 0
{
  const rc = new RepCounter();
  rc.update('Unknown Exercise', makeAngles(0.5));
  assert.equal(rc.get('Unknown Exercise'), 0, 'unknown exercise returns 0');
  console.log('Unknown exercise: PASS');
}

// getCounts returns all tracked exercises
{
  const rc = new RepCounter();
  for (let i = 0; i < 5; i++) rc.update('Deep Squat', makeAngles(0.9));
  for (let i = 0; i < 10; i++) rc.update('Deep Squat', makeAngles(0.2));
  for (let i = 0; i < 15; i++) rc.update('Deep Squat', makeAngles(0.9));
  const counts = rc.getCounts();
  assert.equal(counts['Deep Squat'], 1, 'getCounts returns correct count');
  console.log('getCounts: PASS');
}

console.log('\nAll rep_counter tests passed.');
