const tf = require("./tf");
const fs = require("fs");
const { vocabSize } = require("./tokenizer");
const { getBatch } = require("./data");
const { config, createParams, forward, countParams, saveParams, loadParams } = require("./model");

const batchSize = 32;
const steps = 15000;      // total steps (about 8 hours on your laptop)
const maxLr = 1e-3;       // learning rate at the start
const minLr = 1e-4;       // learning rate at the end
const warmupSteps = 200;  // ramp up gently at the very start
const evalEvery = 250;
const evalBatches = 10;
const latestDir = "checkpoints/latest"; // for resuming
const bestDir = "checkpoints/best";     // lowest val loss so far

const params = createParams(vocabSize);
console.log("parameters:", countParams(params));

// Resume if a checkpoint exists
let startStep = 0;
let bestVal = Infinity;
if (fs.existsSync(`${latestDir}/meta.json`)) {
  const meta = loadParams(params, latestDir);
  startStep = meta.step;
  bestVal = meta.bestVal ?? Infinity;
  console.log(`resumed from step ${startStep} (best val so far ${bestVal.toFixed(3)})`);
}

// Learning rate for a given step: warm up, then a smooth curve down to minLr
function getLr(step) {
  if (step < warmupSteps) return maxLr * (step + 1) / warmupSteps;
  const progress = (step - warmupSteps) / (steps - warmupSteps); // 0 → 1
  return minLr + (maxLr - minLr) * 0.5 * (1 + Math.cos(Math.PI * progress));
}

function computeLoss(xs, ys, training) {
  const B = xs.length, T = xs[0].length;
  const x = tf.tensor1d(xs.flat(), "int32");
  const y = tf.oneHot(tf.tensor1d(ys.flat(), "int32"), vocabSize);
  return tf.losses.softmaxCrossEntropy(y, forward(params, x, B, T, training));
}

// Average loss over several batches, dropout off, no learning
function estimateLoss(split) {
  let total = 0;
  for (let i = 0; i < evalBatches; i++) {
    const { xs, ys } = getBatch(split, batchSize, config.blockSize);
    total += tf.tidy(() => computeLoss(xs, ys, false).dataSync()[0]);
  }
  return total / evalBatches;
}

const optimizer = tf.train.adam(maxLr);
const start = Date.now();

for (let step = startStep; step <= steps; step++) {
  if (step % evalEvery === 0 && step > startStep) {
    const train = estimateLoss("train");
    const val = estimateLoss("val");
    const mins = ((Date.now() - start) / 60000).toFixed(1);
    let note = "";
    if (val < bestVal) {
      bestVal = val;
      saveParams(params, bestDir, step, bestVal);
      note = "  ← new best, saved";
    }
    saveParams(params, latestDir, step, bestVal);
    console.log(`step ${step}  train ${train.toFixed(3)}  val ${val.toFixed(3)}  lr ${getLr(step).toExponential(1)}  ${mins}min${note}`);
  }
  if (step === steps) break;

  optimizer.learningRate = getLr(step);
  const { xs, ys } = getBatch("train", batchSize, config.blockSize);
  tf.tidy(() => { optimizer.minimize(() => computeLoss(xs, ys, true)); });
}

console.log(`done. best val ${bestVal.toFixed(3)}, saved in ${bestDir}`);
