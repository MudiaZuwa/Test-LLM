const tf = require("@tensorflow/tfjs-node");
const { vocabSize, encode, decode } = require("./tokenizer");
const { config, createParams, forward, loadParams } = require("./model");

// Settings (can be changed from the command line)
const temperature = parseFloat(process.argv[2] ?? "0.8"); // lower = safer, higher = wilder
const topK = parseInt(process.argv[3] ?? "0");             // 0 = off
const length = parseInt(process.argv[4] ?? "500");         // characters to write
const prompt = process.argv[5] ?? "\n";                    // starting text

const params = createParams(vocabSize);
const { step } = loadParams(params, "checkpoints/best");
console.log(`loaded checkpoint from step ${step} | temperature ${temperature} | topK ${topK || "off"}\n`);

function sampleNext(ids) {
  return tf.tidy(() => {
    // 1. last blockSize characters
    const context = ids.slice(-config.blockSize);
    const T = context.length;

    // 2. run the model, keep only the last position's scores
    const logits = forward(params, tf.tensor1d(context, "int32"), 1, T);
    let last = logits.slice([T - 1, 0], [1, vocabSize]);

    // 3. temperature: divide scores
    last = last.div(temperature);

    // 4. top-k: keep only the k highest scores
    if (topK > 0) {
      const { values } = tf.topk(last, topK);
      const kth = values.slice([0, topK - 1], [1, 1]); // the k-th best score
      last = tf.where(last.less(kth), tf.fill(last.shape, -1e9), last);
    }

    // 5. softmax + sample (multinomial does both)
    return tf.multinomial(last, 1).dataSync()[0];
  });
}

const ids = encode(prompt);
process.stdout.write(prompt);
for (let i = 0; i < length; i++) {
  const next = sampleNext(ids);
  ids.push(next);
  process.stdout.write(decode([next])); // print each character as it's made
}
console.log();
