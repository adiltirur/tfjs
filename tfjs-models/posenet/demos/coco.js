/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as posenet from '@tensorflow-models/posenet';
import * as tf from '@tensorflow/tfjs';
import dat from 'dat.gui';

import {isMobile, toggleLoadingUI, tryResNetButtonName, tryResNetButtonText, updateTryResNetButtonDatGuiCss} from './demo_util';
// clang-format off
import {
  drawBoundingBox,
  drawKeypoints,
  drawSkeleton,
  renderImageToCanvas,
} from './demo_util';

// clang-format on


const images = [
  'frisbee.jpg',
  'frisbee_2.jpg',
  'backpackman.jpg',
  'boy_doughnut.jpg',
  'soccer.png',
  'with_computer.jpg',
  'snowboard.jpg',
  'person_bench.jpg',
  'skiing.jpg',
  'fire_hydrant.jpg',
  'kyte.jpg',
  'looking_at_computer.jpg',
  'tennis.jpg',
  'tennis_standing.jpg',
  'truck.jpg',
  'on_bus.jpg',
  'tie_with_beer.jpg',
  'baseball.jpg',
  'multi_skiing.jpg',
  'riding_elephant.jpg',
  'skate_park_venice.jpg',
  'skate_park.jpg',
  'tennis_in_crowd.jpg',
  'two_on_bench.jpg',
];

/**
 * Draws a pose if it passes a minimum confidence onto a canvas.
 * Only the pose's keypoints that pass a minPartConfidence are drawn.
 */
function drawResults(canvas, poses, minPartConfidence, minPoseConfidence) {
  renderImageToCanvas(image, [513, 513], canvas);
  const ctx = canvas.getContext('2d');
  var count = 0;
  poses.forEach((pose) => {
    if (pose.score >= minPoseConfidence) {
      count = count+1;
      if (guiState.showKeypoints) {
        drawKeypoints(pose.keypoints, minPartConfidence, ctx);
      }

      if (guiState.showSkeleton) {
        drawSkeleton(pose.keypoints, minPartConfidence, ctx);
      }

      if (guiState.showBoundingBox) {
        drawBoundingBox(pose.keypoints, ctx);
      }
    }
  });
  console.log(count);
}

const imageBucket =
    'https://storage.googleapis.com/tfjs-models/assets/posenet/';

async function loadImage(imagePath) {
  const image = new Image();
  const promise = new Promise((resolve, reject) => {
    image.crossOrigin = '';
    image.onload = () => {
      resolve(image);
    };
  });

  image.src = `${imageBucket}${imagePath}`;
  return promise;
}








function multiPersonCanvas() {
  return document.querySelector('#multi canvas');
}

let image = null;
let predictedPoses = null;

/**
 * Draw the results from the multi-pose estimation on to a canvas
 */
function drawMultiplePosesResults() {
  const canvas = multiPersonCanvas();
  drawResults(
      canvas, predictedPoses, guiState.multiPoseDetection.minPartConfidence,
      guiState.multiPoseDetection.minPoseConfidence);
}

function setStatusText(text) {
  const resultElement = document.getElementById('status');
  resultElement.innerText = text;
}

/**
 * Purges variables and frees up GPU memory using dispose() method
 */
function disposePoses() {
  if (predictedPoses) {
    predictedPoses = null;
  }
}

/**
 * Loads an image, feeds it into posenet the posenet model, and
 * calculates poses based on the model outputs
 */
async function testImageAndEstimatePoses(net) {
  setStatusText('Predicting...');
  document.getElementById('results').style.display = 'none';

  // Purge prevoius variables and free up GPU memory
  disposePoses();

  // Load an example image
  image = await loadImage(guiState.image);



  // Creates a tensor from an image
  const input = tf.browser.fromPixels(image);

  // Estimates poses
  const poses = await net.estimatePoses(input, {
    flipHorizontal: false,
    decodingMethod: 'multi-person',
    maxDetections: guiState.multiPoseDetection.maxDetections,
    scoreThreshold: guiState.multiPoseDetection.minPartConfidence,
    nmsRadius: guiState.multiPoseDetection.nmsRadius
  });
  predictedPoses = poses;

  // Draw poses.
  drawMultiplePosesResults();

  setStatusText('');
  document.getElementById('results').style.display = 'block';
  input.dispose();
}

/**
 * Reloads PoseNet, then loads an image, feeds it into posenet, and
 * calculates poses based on the model outputs
 */
async function reloadNetTestImageAndEstimatePoses(net) {
  if (guiState.net) {
    guiState.net.dispose();
  }
  toggleLoadingUI(true);
  guiState.net = await posenet.load({
    architecture: guiState.model.architecture,
    outputStride: guiState.model.outputStride,
    inputResolution: guiState.model.inputResolution,
    multiplier: guiState.model.multiplier,
    quantBytes: guiState.model.quantBytes,
  });
  toggleLoadingUI(false);
  testImageAndEstimatePoses(guiState.net);
}

const defaultQuantBytes = 2;

const defaultMobileNetMultiplier = isMobile() ? 0.50 : 0.75;
const defaultMobileNetStride = 16;
const defaultMobileNetInputResolution = 513;

const defaultResNetMultiplier = 1.0;
const defaultResNetStride = 32;
const defaultResNetInputResolution = 257;

let guiState = {
  net: null,
  model: {
    architecture: 'MobileNetV1',
    outputStride: defaultMobileNetStride,
    inputResolution: defaultMobileNetInputResolution,
    multiplier: defaultMobileNetMultiplier,
    quantBytes: defaultQuantBytes,
  },
  image: 'tennis_in_crowd.jpg',
  multiPoseDetection: {
    minPartConfidence: 0.1,
    minPoseConfidence: 0.2,
    nmsRadius: 20.0,
    maxDetections: 15,
  },
  showKeypoints: true,
  showSkeleton: true,
  showBoundingBox: false,
};

function setupGui(net) {
  guiState.net = net;
  const gui = new dat.GUI();
};
/**
 * Kicks off the demo by loading the posenet model and estimating
 * poses on a default image
 */
export async function bindPage() {
  toggleLoadingUI(true);
  const net = await posenet.load({
    architecture: guiState.model.architecture,
    outputStride: guiState.model.outputStride,
    inputResolution: guiState.model.inputResolution,
    multiplier: guiState.model.multiplier,
    quantBytes: guiState.model.quantBytes
  });
  toggleLoadingUI(false);
  //setupGui(net); //remove
  await testImageAndEstimatePoses(net);
}

bindPage();
