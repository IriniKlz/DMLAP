/*
  DMLAP 2026
  Image Classification with Teachable Machine
  Reference: https://docs.ml5js.org/#/reference/image-classifier
  and:       https://editor.p5js.org/codingtrain/sketches/PoZXqbu4v 

  First go to https://teachablemachine.withgoogle.com/train/image
  and train our custom image classifier
  Then export the model and copy the URL below
*/

// Replace with your custom model URL here
const imageModelURL = 'https://teachablemachine.withgoogle.com/models/bXy2kDNi/';  // Night vs Day

// Video
let video;

// To store the classification results
let label = "";
let confidence = 0.0;

// Classifier 
let classifier;

async function setup() {
  // Option to load your model or MobileNet
  classifier = await ml5.imageClassifier(imageModelURL + 'model.json');                                                                                                                             
  // classifier = await ml5.imageClassifier("MobileNet");

  createCanvas(320, 260);

  // Create the video
  video = createCapture(VIDEO, { flipped: true }); 
  video.size(320, 240);
  video.hide();

  // Start classifying
  classifyVideo();
}

function draw() {
  background(0);
  // Draw the video
  image(video, 0, 0);

  // Draw the label
  fill(255);
  textSize(16);
  textAlign(CENTER);
  text(label + " w/ " + (confidence * 100).toFixed(2) + "% conf", width / 2, height - 16);

}

// Getting a prediction 
function classifyVideo() {
  classifier.classifyStart(video, gotResult);
}

// Getting results, ie labels and confidence scores
function gotResult(results) {
  // console.log(results[0]);
  label = results[0].label;           
  confidence = results[0].confidence;
}
