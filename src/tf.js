// tfjs-node 4.22 calls util.isNullOrUndefined, which Node 23+ removed
const util = require("util");
if (typeof util.isNullOrUndefined !== "function") {
  util.isNullOrUndefined = (v) => v === null || v === undefined;
}

module.exports = require("@tensorflow/tfjs-node");
