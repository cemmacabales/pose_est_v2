import assert from 'assert/strict';
import { keypointsToAngles, batchKeypointsToAngles, computeMotion } from '../../web/joint_angles.js';

const EPSILON = 1e-5;
function near(a, b) { return Math.abs(a - b) < EPSILON; }
function assertNear(actual, expected, msg) {
  assert(near(actual, expected), `${msg}: got ${actual}, expected ${expected}`);
}

// Test joints: person standing upright, all joints on the vertical axis.
// Joints flat Float32Array: [x0,y0, x1,y1, ...] for 12 joints
// Order: L_sho(0), R_sho(1), L_elb(2), R_elb(3), L_wri(4), R_wri(5),
//        L_hip(6), R_hip(7), L_kne(8), R_kne(9), L_ank(10), R_ank(11)
// Hip midpoint is already at origin; joints are hip-centered.
const uprightJoints = new Float32Array([
  0,  -0.5,   0, -0.5,   // L_sho, R_sho (above)
  0,  -0.3,   0, -0.3,   // L_elb, R_elb
  0,  -0.1,   0, -0.1,   // L_wri, R_wri
  0,   0.1,   0,  0.1,   // L_hip, R_hip
  0,   0.4,   0,  0.4,   // L_kne, R_kne
  0,   0.7,   0,  0.7,   // L_ank, R_ank
]);

// Expected values for uprightJoints:
// ua_L = L_elb - L_sho = [0, 0.2], fa_L = L_wri - L_elb = [0, 0.2]
// [0] L elbow cos([0,0.2],[0,0.2]) = 1.0
// [1] R elbow = 1.0
// ul_L = L_kne - L_hip = [0, 0.3], ll_L = L_ank - L_kne = [0, 0.3]
// [2] L knee cos([0,0.3],[0,0.3]) = 1.0
// [3] R knee = 1.0
// mid_sho = [0,-0.5], torso_unit = [0,-1]
// [4] L arm elevation: cos([0,0.2],[0,-1]) = 0.2*(-1)/(0.2*1) = -1.0
// [5] R arm elevation = -1.0
// [6] L hip flexion: cos([0,0.3],[0,-1]) = -1.0
// [7] R hip flexion = -1.0
// ua_L_unit = unit([0,0.2]) = [0, 1]
// [8-11] lateral x-components = 0
// [12-15] vertical y-components = 1

{
  const angles = keypointsToAngles(uprightJoints);
  assert.equal(angles.length, 16, 'output length is 16');
  assertNear(angles[0],   1.0, 'L elbow straight');
  assertNear(angles[1],   1.0, 'R elbow straight');
  assertNear(angles[2],   1.0, 'L knee straight');
  assertNear(angles[3],   1.0, 'R knee straight');
  assertNear(angles[4],  -1.0, 'L arm elevation (down vs up torso)');
  assertNear(angles[5],  -1.0, 'R arm elevation');
  assertNear(angles[6],  -1.0, 'L hip flexion');
  assertNear(angles[7],  -1.0, 'R hip flexion');
  assertNear(angles[8],   0.0, 'L arm lateral x');
  assertNear(angles[9],   0.0, 'R arm lateral x');
  assertNear(angles[10],  0.0, 'L leg lateral x');
  assertNear(angles[11],  0.0, 'R leg lateral x');
  assertNear(angles[12],  1.0, 'L arm vertical y');
  assertNear(angles[13],  1.0, 'R arm vertical y');
  assertNear(angles[14],  1.0, 'L leg vertical y');
  assertNear(angles[15],  1.0, 'R leg vertical y');
  console.log('keypointsToAngles: PASS');
}

// batchKeypointsToAngles: 3 identical frames should give the same angles each
{
  const batchJoints = new Float32Array(3 * 24);
  for (let f = 0; f < 3; f++) batchJoints.set(uprightJoints, f * 24);
  const batchAngles = batchKeypointsToAngles(batchJoints, 3);
  assert.equal(batchAngles.length, 3 * 16, 'batch output length');
  for (let f = 0; f < 3; f++) {
    for (let i = 0; i < 16; i++) {
      assertNear(batchAngles[f * 16 + i], i < 4 ? 1.0 : (i < 8 ? -1.0 : (i < 12 ? 0.0 : 1.0)),
        `batch frame ${f} feature ${i}`);
    }
  }
  console.log('batchKeypointsToAngles: PASS');
}

// computeMotion: identical frames → motion = 0
{
  const batchJoints = new Float32Array(30 * 24);
  for (let f = 0; f < 30; f++) batchJoints.set(uprightJoints, f * 24);
  const angles = batchKeypointsToAngles(batchJoints, 30);
  const motion = computeMotion(angles, 30);
  assertNear(motion, 0.0, 'motion for identical frames');
  console.log('computeMotion identical frames: PASS');
}

// computeMotion: varying frames → motion > 0
{
  const batchJoints = new Float32Array(30 * 24);
  for (let f = 0; f < 30; f++) {
    const frame = new Float32Array(uprightJoints);
    // vary knee position each frame
    frame[16] = 0.4 + f * 0.01; // L_kne y
    frame[18] = 0.4 + f * 0.01; // R_kne y
    batchJoints.set(frame, f * 24);
  }
  const angles = batchKeypointsToAngles(batchJoints, 30);
  const motion = computeMotion(angles, 30);
  assert(motion > 0, `motion should be > 0, got ${motion}`);
  console.log('computeMotion varying frames: PASS');
}

console.log('\nAll joint_angles tests passed.');
