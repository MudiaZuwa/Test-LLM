const fs = require("fs");

const text = fs.readFileSync("data/input.txt", "utf8");

// TODO 1: get the unique characters, sorted
// hint: new Set(text) gives unique chars, [...set] makes an array, .sort()
const chars = [...new Set(text)].sort();

const vocabSize = chars.length;

const charToId = {};
const idToChar = {};
chars.forEach((ch, i) => {
  charToId[ch] = i;
  idToChar[i] = ch;
});

// TODO 2: string → array of numbers
// hint: [...str].map(...)
function encode(str) {
  return [...str].map(ch => charToId[ch]);
}

// TODO 3: array of numbers → string
// hint: .map(...) then .join("")
function decode(ids) {
  return ids.map(id => idToChar[id]).join("");
}

module.exports = { text, chars, vocabSize, encode, decode };

// quick test (only runs when you run this file directly)
if (require.main === module) {
  console.log("vocab size:", vocabSize);
  console.log(encode("hello"));
  console.log(decode(encode("hello")));
}

