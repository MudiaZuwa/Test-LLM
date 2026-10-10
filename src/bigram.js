const tf = require("@tensorflow/tfjs");
const { vocabSize, decode } = require("./tokenizer");
const { getBatch } = require("./data");

const batchSize = 32;
const blockSize = 8;
const steps = 3000;
const lr = 1e-2; // learning rate: how big each nudge is

// The model: a 65x65 table of scores, starting as small random numbers
const table = tf.variable(tf.randomNormal([vocabSize, vocabSize], 0, 0.01));

// Look up each character's row in the table → scores for the next char
function forward(ids) {
  const oneHot = tf.oneHot(ids, vocabSize).toFloat();
  return tf.matMul(oneHot, table);
}

// How wrong are we?
function computeLoss(xs, ys) {
  const x = tf.tensor1d(xs.flat(), "int32");
  const y = tf.oneHot(tf.tensor1d(ys.flat(), "int32"), vocabSize);
  return tf.losses.softmaxCrossEntropy(y, forward(x));
}

const optimizer = tf.train.adam(lr);

// Training loop
for (let step = 0; step <= steps; step++) {
  const { xs, ys } = getBatch("train", batchSize, blockSize);

  // minimize = compute loss, work out the nudges, apply them
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

// Write new text, one character at a time
function generate(length) {
  const ids = [0]; // start with id 0, which is a newline
  for (let i = 0; i < length; i++) {
    const next = tf.tidy(() => {
      const last = tf.tensor1d([ids[ids.length - 1]], "int32");
      const logits = forward(last);
      return tf.multinomial(logits, 1).dataSync()[0]; // pick randomly, weighted by score
    });
    ids.push(next);
  }
  return decode(ids);
}

console.log(generate(300));
