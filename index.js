/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import {IMAGE_H, IMAGE_W, MnistData} from './data';
import * as ui from './ui';

function createDenseModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({inputShape: [IMAGE_H, IMAGE_W, 1]}));
  model.add(tf.layers.dense({units: ui.getModelDenseSize(), activation: 'relu'}));
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
  return model;
}

async function train(model, onIteration) {
  ui.logStatus('Training model...');

  const learningRate = ui.getTrainLearningRate();
  const batchSize = ui.getTrainBatchSize();
  const trainEpochs = ui.getTrainEpochs();

  const validationSplit = 0.15;
  let trainBatchCount = 0;
  const trainData = data.getTrainData();
  const testData = data.getTestData();

  const totalNumBatches =
      Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) *
      trainEpochs;
  let startTime = (new Date()).getTime();

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  let valAcc;
  await model.fit(trainData.xs, trainData.labels, {
    batchSize,
    validationSplit,
    epochs: trainEpochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        trainBatchCount++;
        ui.logStatus(
            `Training... (` +
            `${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%` +
            ` complete) in ${((new Date()).getTime() - startTime) / 1000} seconds. To stop training, refresh or close page.`);
        ui.plotLoss(trainBatchCount, logs.loss, 'train');
        ui.plotAccuracy(trainBatchCount, logs.acc, 'train');
        if (onIteration && batch % 10 === 0) {
          onIteration('onBatchEnd', batch, logs);
        }
        await tf.nextFrame();
      },
      onEpochEnd: async (epoch, logs) => {
        valAcc = logs.val_acc;
        ui.plotLoss(trainBatchCount, logs.val_loss, 'validation');
        ui.plotAccuracy(trainBatchCount, logs.val_acc, 'validation');
        if (onIteration) {
          onIteration('onEpochEnd', epoch, logs);
        }
        await tf.nextFrame();
      }
    }
  });

  const testResult = model.evaluate(testData.xs, testData.labels);
  const testAccPercent = testResult[1].dataSync()[0] * 100;
  const finalValAccPercent = valAcc * 100;
  ui.logStatus(
      `Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
      `Final test accuracy: ${testAccPercent.toFixed(1)}%;` +
      `Train time: ${((new Date()).getTime() - startTime) / 1000} seconds`);
}

async function showPredictions(model) {
  const testExamples = 100;
  const examples = data.getTestData(testExamples);

  tf.tidy(() => {
    const output = model.predict(examples.xs);
    const axis = 1;
    const labels = Array.from(examples.labels.argMax(axis).dataSync());
    const predictions = Array.from(output.argMax(axis).dataSync());

    ui.showTestResults(examples, predictions, labels);
  });
}

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

ui.setTrainButtonCallback(async () => {
  ui.logStatus('Loading MNIST data...');
  await load();

  ui.logStatus('Creating model...');
  const model = createDenseModel();
  model.summary();

  ui.logStatus('Starting model training...');
  await train(model, (step, stepInfo, logs) => {
    console.log(`${step}: ${stepInfo} -> loss:${logs.loss}, acc:${logs.acc}`);
    showPredictions(model)
  });
});
