const tf = require("./tf");

const { vocabSize, decode } = require("./tokenizer");
const { getBatch } = require("./data");

const batchSize = 32;
const blockSize = 32; // how far back it can look
const embDim = 32;    // numbers per character description
const steps = 3000;
const lr = 3e-3;

async function main() {

  // ---- learnable pieces ----
  const init = (shape) => tf.variable(tf.randomNormal(shape, 0, 0.02));
  const tokEmb = init([vocabSize, embDim]);  // description of each character
  const posEmb = init([blockSize, embDim]);  // description of each position
  const Wq = init([embDim, embDim]);         // makes queries
  const Wk = init([embDim, embDim]);         // makes keys
  const Wv = init([embDim, embDim]);         // makes values
  const Wout = init([embDim, vocabSize]);    // turns result into 65 scores
  const bOut = tf.variable(tf.zeros([vocabSize]));

  // ---- the model ----
  function forward(ids, B, T) {
    // 1. character description + position description
    const tok = tf.matMul(tf.oneHot(ids, vocabSize).toFloat(), tokEmb);
    const pos = posEmb.slice([0, 0], [T, embDim]);
    const x = tok.reshape([B, T, embDim]).add(pos);

    // 2. make query, key, value for every character
    const x2d = x.reshape([B * T, embDim]);
    const q = tf.matMul(x2d, Wq).reshape([B, T, embDim]);
    const k = tf.matMul(x2d, Wk).reshape([B, T, embDim]);
    const v = tf.matMul(x2d, Wv).reshape([B, T, embDim]);

    // 3. scores: how well each query matches each key
    let scores = tf.matMul(q, k, false, true).div(Math.sqrt(embDim));

    // 4. causal mask: block the future
    const lower = tf.linalg.bandPart(tf.ones([T, T]), -1, 0);
    const mask = tf.sub(1, lower).mul(-1e9);
    scores = scores.add(mask);

    // 5. scores → percentages → mix the values
    const weights = tf.softmax(scores);
    const out = tf.matMul(weights, v);

    // 6. turn into 65 next-character scores
    return tf.matMul(out.reshape([B * T, embDim]), Wout).add(bOut);
  }

  function computeLoss(xs, ys) {
    const B = xs.length, T = xs[0].length;
    const x = tf.tensor1d(xs.flat(), "int32");
    const y = tf.oneHot(tf.tensor1d(ys.flat(), "int32"), vocabSize);
    return tf.losses.softmaxCrossEntropy(y, forward(x, B, T));
  }

  // ---- training (same as bigram) ----
  const optimizer = tf.train.adam(lr);
  for (let step = 0; step <= steps; step++) {
    const { xs, ys } = getBatch("train", batchSize, blockSize);
    const loss = tf.tidy(() => optimizer.minimize(() => computeLoss(xs, ys), true));

    if (step % 500 === 0) {
      const val = tf.tidy(() => {
        const b = getBatch("val", batchSize, blockSize);
        return computeLoss(b.xs, b.ys);
      });
      console.log(`step ${step}  train ${loss.dataSync()[0].toFixed(3)}  val ${val.dataSync()[0].toFixed(3)}`);
      val.dispose();
    }
    loss.dispose();
  }

  // ---- generate ----
  function generate(length) {
    const ids = [0];
    for (let i = 0; i < length; i++) {
      const next = tf.tidy(() => {
        const context = ids.slice(-blockSize); // last 32 characters
        const T = context.length;
        const logits = forward(tf.tensor1d(context, "int32"), 1, T);
        const last = logits.slice([T - 1, 0], [1, vocabSize]); // only the newest position
        return tf.multinomial(last, 1).dataSync()[0];
      });
      ids.push(next);
    }
    return decode(ids);
  }

  console.log(generate(300));
}

main();
