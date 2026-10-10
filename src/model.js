const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");

const config = {
  blockSize: 128, // how far back it can look
  embDim: 128,   // numbers per character note
  nHeads: 4,     // attention heads per block
  nLayers: 3,    // number of blocks stacked
  dropout: 0.1,  // fraction of numbers switched off during training
};

// ---- create all learnable numbers, stored by name ----
function createParams(vocabSize) {
  const { blockSize, embDim, nLayers } = config;
  const params = {};
  const rand = (name, shape) => (params[name] = tf.variable(tf.randomNormal(shape, 0, 0.02), true, name));
  const zeros = (name, shape) => (params[name] = tf.variable(tf.zeros(shape), true, name));
  const ones = (name, shape) => (params[name] = tf.variable(tf.ones(shape), true, name));

  rand("tokEmb", [vocabSize, embDim]);
  rand("posEmb", [blockSize, embDim]);

  for (let l = 0; l < nLayers; l++) {
    const p = `layer${l}.`;
    // attention part
    ones(p + "ln1.gamma", [embDim]);
    zeros(p + "ln1.beta", [embDim]);
    rand(p + "Wq", [embDim, embDim]);
    rand(p + "Wk", [embDim, embDim]);
    rand(p + "Wv", [embDim, embDim]);
    rand(p + "Wproj", [embDim, embDim]);
    zeros(p + "bproj", [embDim]);
    // feed-forward part
    ones(p + "ln2.gamma", [embDim]);
    zeros(p + "ln2.beta", [embDim]);
    rand(p + "W1", [embDim, 4 * embDim]);
    zeros(p + "b1", [4 * embDim]);
    rand(p + "W2", [4 * embDim, embDim]);
    zeros(p + "b2", [embDim]);
  }

  ones("lnF.gamma", [embDim]);
  zeros("lnF.beta", [embDim]);
  rand("Wout", [embDim, vocabSize]);
  zeros("bOut", [vocabSize]);
  return params;
}

// ---- building blocks ----

function layerNorm(x, gamma, beta) {
  const { mean, variance } = tf.moments(x, -1, true);
  return x.sub(mean).div(variance.add(1e-5).sqrt()).mul(gamma).add(beta);
}

function multiHeadAttention(x, params, p, B, T) {
  const { embDim, nHeads } = config;
  const headSize = embDim / nHeads; // 128 / 4 = 32 numbers per head

  // cut each note into 4 pieces, one per head
  const split = (t) =>
    t.reshape([B, T, nHeads, headSize]).transpose([0, 2, 1, 3]).reshape([B * nHeads, T, headSize]);

  const q = split(tf.matMul(x, params[p + "Wq"]));
  const k = split(tf.matMul(x, params[p + "Wk"]));
  const v = split(tf.matMul(x, params[p + "Wv"]));

  // same attention as Step 5, but all heads at once
  let scores = tf.matMul(q, k, false, true).div(Math.sqrt(headSize));
  const lower = tf.linalg.bandPart(tf.ones([T, T]), -1, 0);
  scores = scores.add(tf.sub(1, lower).mul(-1e9));
  const out = tf.matMul(tf.softmax(scores), v);

  // glue the 4 heads back together, then mix them
  const joined = out.reshape([B, nHeads, T, headSize]).transpose([0, 2, 1, 3]).reshape([B * T, embDim]);
  return tf.matMul(joined, params[p + "Wproj"]).add(params[p + "bproj"]);
}

function feedForward(x, params, p) {
  const h = tf.relu(tf.matMul(x, params[p + "W1"]).add(params[p + "b1"]));
  return tf.matMul(h, params[p + "W2"]).add(params[p + "b2"]);
}

// ---- the full model ----
function forward(params, ids, B, T, training = false) {
  const { embDim, nLayers } = config;
  const drop = (t) => (training ? tf.dropout(t, config.dropout) : t);
  const vocabSize = params.tokEmb.shape[0];

  // notes: what + where
  const tok = tf.matMul(tf.oneHot(ids, vocabSize).toFloat(), params.tokEmb);
  const pos = params.posEmb.slice([0, 0], [T, embDim]);
  let x = drop(tok.reshape([B, T, embDim]).add(pos).reshape([B * T, embDim]));

  // the stacked blocks
  for (let l = 0; l < nLayers; l++) {
    const p = `layer${l}.`;
    x = x.add(drop(multiHeadAttention(layerNorm(x, params[p + "ln1.gamma"], params[p + "ln1.beta"]), params, p, B, T)));
    x = x.add(drop(feedForward(layerNorm(x, params[p + "ln2.gamma"], params[p + "ln2.beta"]), params, p)));
  }

  // final tidy-up → 65 scores
  x = layerNorm(x, params["lnF.gamma"], params["lnF.beta"]);
  return tf.matMul(x, params.Wout).add(params.bOut);
}

function countParams(params) {
  return Object.values(params).reduce((sum, v) => sum + v.size, 0);
}

// ---- save / load ----
// Writes two files: weights.bin (all the numbers) and meta.json (names, shapes, step)
function saveParams(params, dir, step, bestVal) {
  fs.mkdirSync(dir, { recursive: true });
  const meta = { step, bestVal, config, tensors: [] };
  const chunks = [];
  for (const [name, v] of Object.entries(params)) {
    meta.tensors.push({ name, shape: v.shape });
    chunks.push(Buffer.from(v.dataSync().buffer));
  }
  fs.writeFileSync(`${dir}/weights.bin`, Buffer.concat(chunks));
  fs.writeFileSync(`${dir}/meta.json`, JSON.stringify(meta, null, 2));
}

// Fills existing params with saved numbers. Returns the saved meta (step, bestVal).
function loadParams(params, dir) {
  const meta = JSON.parse(fs.readFileSync(`${dir}/meta.json`, "utf8"));
  const buf = fs.readFileSync(`${dir}/weights.bin`);
  const all = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  let offset = 0;
  for (const { name, shape } of meta.tensors) {
    const size = shape.reduce((a, b) => a * b, 1);
    params[name].assign(tf.tensor(all.slice(offset, offset + size), shape));
    offset += size;
  }
  return meta;
}

module.exports = { config, createParams, forward, countParams, saveParams, loadParams };
