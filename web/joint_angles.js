// joint_angles.js — port of joint_angles.py
// Joints are flat Float32Array: [x0,y0, x1,y1, ...] for 12 joints.
// Ordering: L_sho, R_sho, L_elb, R_elb, L_wri, R_wri,
//           L_hip, R_hip, L_kne, R_kne, L_ank, R_ank

function _norm2(x, y) { return Math.sqrt(x * x + y * y); }

function _unitX(x, y) { const n = _norm2(x, y); return n < 1e-8 ? 0 : x / n; }
function _unitY(x, y) { const n = _norm2(x, y); return n < 1e-8 ? 0 : y / n; }

function _cos(ax, ay, bx, by) {
  const na = _norm2(ax, ay);
  const nb = _norm2(bx, by);
  if (na < 1e-8 || nb < 1e-8) return 1.0;
  return Math.max(-1.0, Math.min(1.0, (ax * bx + ay * by) / (na * nb)));
}

export function keypointsToAngles(joints) {
  // joints: Float32Array length 24 — (12 joints × 2 coords), hip-centered
  const g = (i) => [joints[i * 2], joints[i * 2 + 1]];

  const [lsx, lsy] = g(0),  [rsx, rsy] = g(1);
  const [lex, ley] = g(2),  [rex, rey] = g(3);
  const [lwx, lwy] = g(4),  [rwx, rwy] = g(5);
  const [lhx, lhy] = g(6),  [rhx, rhy] = g(7);
  const [lkx, lky] = g(8),  [rkx, rky] = g(9);
  const [lax, lay] = g(10), [rax, ray] = g(11);

  const uaLx = lex - lsx, uaLy = ley - lsy;  // upper arm L
  const uaRx = rex - rsx, uaRy = rey - rsy;  // upper arm R
  const faLx = lwx - lex, faLy = lwy - ley;  // forearm L
  const faRx = rwx - rex, faRy = rwy - rey;  // forearm R
  const ulLx = lkx - lhx, ulLy = lky - lhy; // upper leg L
  const ulRx = rkx - rhx, ulRy = rky - rhy; // upper leg R
  const llLx = lax - lkx, llLy = lay - lky; // lower leg L
  const llRx = rax - rkx, llRy = ray - rky; // lower leg R

  const torsoX = (lsx + rsx) * 0.5;  // mid_shoulder; hip midpoint is origin
  const torsoY = (lsy + rsy) * 0.5;
  const tuX = _unitX(torsoX, torsoY);
  const tuY = _unitY(torsoX, torsoY);

  return new Float32Array([
    _cos(uaLx, uaLy, faLx, faLy),      // 0  L elbow
    _cos(uaRx, uaRy, faRx, faRy),      // 1  R elbow
    _cos(ulLx, ulLy, llLx, llLy),      // 2  L knee
    _cos(ulRx, ulRy, llRx, llRy),      // 3  R knee
    _cos(uaLx, uaLy, tuX, tuY),        // 4  L arm elevation
    _cos(uaRx, uaRy, tuX, tuY),        // 5  R arm elevation
    _cos(ulLx, ulLy, tuX, tuY),        // 6  L hip flexion
    _cos(ulRx, ulRy, tuX, tuY),        // 7  R hip flexion
    _unitX(uaLx, uaLy),                // 8  L arm x
    _unitX(uaRx, uaRy),                // 9  R arm x
    _unitX(ulLx, ulLy),                // 10 L leg x
    _unitX(ulRx, ulRy),                // 11 R leg x
    _unitY(uaLx, uaLy),                // 12 L arm y
    _unitY(uaRx, uaRy),                // 13 R arm y
    _unitY(ulLx, ulLy),                // 14 L leg y
    _unitY(ulRx, ulRy),                // 15 R leg y
  ]);
}

export function batchKeypointsToAngles(batchJoints, numFrames) {
  // batchJoints: Float32Array length numFrames * 24
  // Returns Float32Array length numFrames * 16
  const out = new Float32Array(numFrames * 16);
  for (let f = 0; f < numFrames; f++) {
    const frame = batchJoints.subarray(f * 24, (f + 1) * 24);
    out.set(keypointsToAngles(frame), f * 16);
  }
  return out;
}

export function computeMotion(angles, numFrames) {
  // angles: Float32Array length numFrames * 16
  // Returns mean per-feature population std across frames (matches numpy np.std)
  if (numFrames <= 1) return 0;
  let total = 0;
  for (let feat = 0; feat < 16; feat++) {
    let sum = 0;
    for (let f = 0; f < numFrames; f++) sum += angles[f * 16 + feat];
    const mean = sum / numFrames;
    let variance = 0;
    for (let f = 0; f < numFrames; f++) {
      const d = angles[f * 16 + feat] - mean;
      variance += d * d;
    }
    total += Math.sqrt(variance / numFrames);
  }
  return total / 16;
}
