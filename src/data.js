const { text, encode, decode } = require("./tokenizer");

const blockSize = 8;   // characters per chunk (small for now so we can read it)
const batchSize = 4;   // chunks per batch

const data = encode(text);

// TODO 1: split 90/10
// hint: Math.floor(data.length * 0.9), then data.slice(...)
const n = Math.floor(data.length * 0.9);
const trainData = data.slice(0, n);
const valData = data.slice(n);

function getBatch(split, bs = batchSize, bl = blockSize) {
  const d = split === "train" ? trainData : valData;
  const xs = [];
  const ys = [];
  for (let b = 0; b < bs; b++) {
    // TODO 2: random start position
    // hint: it must leave room for blockSize + 1 characters
    const i = Math.floor(Math.random() * (d.length - bl - 1));

    // TODO 3: take the input chunk, and the same chunk shifted by 1
    xs.push(d.slice(i, i + bl));
    ys.push(d.slice(i + 1, i + bl + 1));
  }
  return { xs, ys };
}

module.exports = { trainData, valData, getBatch, blockSize, batchSize };

if (require.main === module) {
  console.log("train:", trainData.length, "val:", valData.length);
  const { xs, ys } = getBatch("train");
  for (let b = 0; b < batchSize; b++) {
    console.log(JSON.stringify(decode(xs[b])), "→", JSON.stringify(decode(ys[b])));
  }
}
