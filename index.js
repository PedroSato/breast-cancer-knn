require("@tensorflow/tfjs-node-gpu");
const tf = require("@tensorflow/tfjs");
const loadCsv = require("./load-csv");

const standardizeData = (tensor, mean, variance) => {
  return tensor.sub(mean).div(variance.pow(0.5));
};

const knn = (features, labels, predictionPoint, k) => {
  const { mean, variance } = tf.moments(features, 0);

  const scaledPrediction = standardizeData(predictionPoint, mean, variance);

  const standarizedFeatures = standardizeData(features, mean, variance);

  return standarizedFeatures
    .sub(scaledPrediction)
    .pow(2)
    .sum(1)
    .pow(0.5)
    .expandDims(1)
    .concat(labels, 1)
    .unstack()
    .sort((a, b) => (a.get(0) > b.get(0) ? 1 : -1))
    .slice(0, k)
    .reduce(
      (acc, pair) => {
        acc[pair.get(1)]++;
        return acc;
      },
      { 2: 0, 4: 0 }
    );
  // .reduce((acc, pair) => acc + pair.get(1), 0) / k
};

let { features, labels, testFeatures, testLabels } = loadCsv(
  "brest_cancer.csv",
  {
    shuffle: true,
    splitTest: 50,
    dataColumns: [
      "clump_thickness",
      "uniformity_size",
      "uniformity_shape",
      "marginal_adhesion",
      "single_size",
      "bare_nuclei",
      "bland_chromatin",
      "normal_nucleoli",
      "mitoses",
    ],
    labelColumns: ["class"],
  }
);

const convertIntoTensor = (array) => tf.tensor(array);

let hits = 0;

testFeatures.forEach((testPoint, i) => {
  const result = knn(
    convertIntoTensor(features),
    convertIntoTensor(labels),
    convertIntoTensor(testPoint),
    10
  );
  let maxCount = Math.max(...Object.values(result));
  let mostFrequent = Object.keys(result).filter(
    (key) => result[key] === maxCount
  );

  const rightOrWrong = testLabels[i][0] - mostFrequent === 0 ? true : false;
  if (rightOrWrong) hits++;

  console.log(
    "the tumor:",
    testFeatures[i],
    "was classified as",
    mostFrequent[0] === "2" ? "benign" : "malignant",
    "and it was actually",
    testLabels[i][0] === 2 ? "benign" : "malignant",
    rightOrWrong
      ? "\u001b[" + 32 + "m" + "✓" + "\u001b[0m"
      : "\u001b[" + 31 + "m" + "X" + "\u001b[0m"
  );

});
console.log("Precision of", (hits / testFeatures.length) * 100 + "%");
