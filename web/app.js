// app.js
import { batchKeypointsToAngles, computeMotion, keypointsToAngles } from './joint_angles.js';
import { RepCounter } from './rep_counter.js';
import { PoseLandmarker, FilesetResolver, DrawingUtils } from
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs';

// ── constants ────────────────────────────────────────────────────────────────
const MAPPED_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const EXERCISE_NAMES = {
  0: 'Deep Squat', 1: 'Hurdle Step', 2: 'Inline Lunge', 3: 'Side Lunge',
  4: 'Sit to Stand', 5: 'Standing Leg Raise',
  6: 'Shoulder Abduction', 7: 'Shoulder Extension', 8: 'Shoulder Scaption',
};
const WINDOW_SIZE = 30;
const IDLE_THRESHOLD = 0.03;
const IDLE_CONFIRM = 2;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const video     = document.getElementById('video');
const canvas    = document.getElementById('canvas');
const ctx       = canvas.getContext('2d');
const statusEl  = document.getElementById('status');
const exerciseEl = document.getElementById('exercise-val');
const qualityEl  = document.getElementById('quality-badge');
const repsEl     = document.getElementById('reps-val');
const confFill   = document.getElementById('conf-fill');
const confPct    = document.getElementById('conf-pct');
const keypointsEl = document.getElementById('keypoints-val');

// ── state ────────────────────────────────────────────────────────────────────
const frameBuffer = [];        // Float32Array(24) per frame, max WINDOW_SIZE
const predBuffer  = [];        // {exIdx, qualIdx, conf}, max 10
let idleCount = 0;
let frameCounter = 0;
let currentExercise = null;
const repCounter = new RepCounter();
let classifying = false;       // debounce: one classify() at a time

// ── models ───────────────────────────────────────────────────────────────────
let poseLandmarker = null;
let tfliteModel    = null;
let exerciseOutIdx = -1;       // index in model.outputs with shape [..., 9]
let qualityOutIdx  = -1;       // index in model.outputs with shape [..., 2]
let drawingUtils   = null;

async function initModels() {
  statusEl.textContent = 'Loading MediaPipe…';
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numPoses: 1,
  });
  drawingUtils = new DrawingUtils(ctx);

  statusEl.textContent = 'Loading classifier…';
  tfliteModel = await tf.loadLayersModel('./models/classifier_tfjs/model.json');

  statusEl.textContent = 'Requesting camera…';
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  await new Promise(resolve => video.addEventListener('loadeddata', resolve, { once: true }));
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  statusEl.textContent = 'Running';
}

// ── joint extraction ─────────────────────────────────────────────────────────
function extractJoints(landmarks) {
  // landmarks: array of 33 {x, y, z, visibility} objects
  const hipMidX = (landmarks[23].x + landmarks[24].x) / 2;
  const hipMidY = (landmarks[23].y + landmarks[24].y) / 2;
  const joints = new Float32Array(24);
  MAPPED_INDICES.forEach((idx, i) => {
    joints[i * 2]     = landmarks[idx].x - hipMidX;
    joints[i * 2 + 1] = landmarks[idx].y - hipMidY;
  });
  return joints;
}

// ── classifier ───────────────────────────────────────────────────────────────
async function classify(angles) {
  if (classifying || !tfliteModel) return;
  classifying = true;
  let outputList = null;
  try {
    const inputData = new Float32Array(1 * WINDOW_SIZE * 16);
    inputData.set(angles);
    const input = tf.tensor(inputData, [1, WINDOW_SIZE, 16]);
    const rawOutputs = tfliteModel.predict(input);
    input.dispose();

    outputList = Array.isArray(rawOutputs) ? rawOutputs : Object.values(rawOutputs);

    let exTensor = exerciseOutIdx >= 0 ? outputList[exerciseOutIdx] : null;
    let qualTensor = qualityOutIdx >= 0 ? outputList[qualityOutIdx] : null;
    if (!exTensor || !qualTensor) {
      for (const t of outputList) {
        const last = t.shape[t.shape.length - 1];
        if (last === 9) exTensor = t;
        else if (last === 2) qualTensor = t;
      }
    }
    if (!exTensor || !qualTensor) throw new Error('Model output shape mismatch');

    const exProbs   = await exTensor.data();
    const qualProbs = await qualTensor.data();

    let exIdx = 0, maxP = -1;
    exProbs.forEach((p, i) => { if (p > maxP) { maxP = p; exIdx = i; } });
    const qualIdx = qualProbs[1] > qualProbs[0] ? 1 : 0;
    const conf = maxP;

    const name = EXERCISE_NAMES[exIdx];
    if (name) currentExercise = name;

    if (predBuffer.length >= 10) predBuffer.shift();
    predBuffer.push({ exIdx, qualIdx, conf });

    if (predBuffer.length >= 5) updateUI();
  } catch (err) {
    console.error('classify error:', err);
  } finally {
    if (outputList) outputList.forEach(t => { try { t.dispose(); } catch (_) {} });
    classifying = false;
  }
}

// ── UI update ────────────────────────────────────────────────────────────────
function updateUI() {
  const exIndices = predBuffer.map(p => p.exIdx);
  const stableEx  = exIndices.sort((a,b) =>
    exIndices.filter(v=>v===b).length - exIndices.filter(v=>v===a).length
  )[0];
  const stableQual = predBuffer.filter(p => p.qualIdx === 1).length > predBuffer.length / 2 ? 1 : 0;
  const stableConf = predBuffer.reduce((s, p) => s + p.conf, 0) / predBuffer.length;

  exerciseEl.textContent = EXERCISE_NAMES[stableEx] ?? 'Unknown';

  qualityEl.className = '';
  if (stableQual === 1) {
    qualityEl.textContent = 'CORRECT';
    qualityEl.classList.add('correct');
  } else {
    qualityEl.textContent = 'INCORRECT';
    qualityEl.classList.add('incorrect');
  }

  const pct = Math.round(stableConf * 100);
  confFill.style.width = pct + '%';
  confFill.style.background = stableConf >= 0.75 ? '#69F0AE' : stableConf >= 0.5 ? '#FFD740' : '#FF6E40';
  confPct.textContent = pct + '%';

  const repName = EXERCISE_NAMES[stableEx];
  if (repName) repsEl.textContent = String(repCounter.get(repName));
}

function showIdle() {
  exerciseEl.textContent = 'Idle';
  qualityEl.textContent  = 'WAITING';
  qualityEl.className    = '';
  confFill.style.width   = '0%';
  confPct.textContent    = '0%';
  repsEl.textContent     = '—';
}

// ── main loop ────────────────────────────────────────────────────────────────
let lastTimestamp = -1;

function loop(timestampMs) {
  requestAnimationFrame(loop);

  if (!poseLandmarker || video.readyState < 2) return;
  if (timestampMs === lastTimestamp) return;
  lastTimestamp = timestampMs;

  // Draw video frame to canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const result = poseLandmarker.detectForVideo(video, timestampMs);
  const landmarks = result.landmarks[0] ?? null;

  // Draw skeleton overlay
  if (landmarks) {
    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS,
      { color: '#00FF00', lineWidth: 2 });
    drawingUtils.drawLandmarks(landmarks,
      { color: '#00FF00', lineWidth: 1, radius: 3 });
  }

  // Keypoints count
  const visible = landmarks
    ? landmarks.filter(lm => (lm.visibility ?? 0) >= 0.5).length
    : 0;
  keypointsEl.textContent = `${visible} / 33 detected`;

  if (landmarks) {
    const joints = extractJoints(landmarks);

    if (frameBuffer.length >= WINDOW_SIZE) frameBuffer.shift();
    frameBuffer.push(joints);

    // Per-frame rep counting
    if (currentExercise) {
      repCounter.update(currentExercise, keypointsToAngles(joints));
    }
  }

  frameCounter++;

  if (frameBuffer.length === WINDOW_SIZE && frameCounter % 5 === 0) {
    const flatBatch = new Float32Array(WINDOW_SIZE * 24);
    frameBuffer.forEach((f, i) => flatBatch.set(f, i * 24));
    const angles = batchKeypointsToAngles(flatBatch, WINDOW_SIZE);
    const motion = computeMotion(angles, WINDOW_SIZE);

    const wasIdle = idleCount >= IDLE_CONFIRM;
    idleCount = motion < IDLE_THRESHOLD
      ? Math.min(idleCount + 1, IDLE_CONFIRM)
      : 0;
    const isIdle = idleCount >= IDLE_CONFIRM;

    if (isIdle) {
      if (!wasIdle) {
        predBuffer.length = 0;
        currentExercise = null;
      }
      showIdle();
    } else {
      classify(angles);  // async, non-blocking
    }
  }
}

// ── boot ─────────────────────────────────────────────────────────────────────
(async () => {
  // Camera and models initialize in parallel; loop starts once camera is ready.
  // Model failures are logged but don't block the camera feed.
  const modelP = initModels().catch(err => {
    console.error('Model init failed:', err);
    statusEl.textContent = `Model error: ${err.message}`;
  });
  const cameraP = startCamera().catch(err => {
    console.error('Camera init failed:', err);
    statusEl.textContent = `Camera error: ${err.message}`;
  });
  await cameraP;
  requestAnimationFrame(loop);
  await modelP; // let model finish loading in background
})();
