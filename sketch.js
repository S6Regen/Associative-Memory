// Vector to vector associative memory using a Locality Sensitive Hash (LSH.)
class AM {
  // vecLen must be 2,4,8,16,32.....
  constructor(vecLen, density, hash) {
    this.vecLen = vecLen;
    this.density = density;
    this.hash = hash;
    this.weights = new Float32Array(vecLen * density);
    this.lsh = new Int8Array(vecLen * density); //store locality sensitive hash
    this.workA = new Float32Array(vecLen);
    this.workB = new Float32Array(vecLen);
  }

  train(target, input) {
    this.recallLSH(this.workB, input); //Recall and store LSH
    subtractVec(this.workB, target, this.workB); // error vector
    scaleVec(this.workB, this.workB, 1 / this.density); //scale correctly before distributing over the weights
    let wtIdx = 0;
    for (let i = 0; i < this.density; i++) {
      for (let j = 0; j < this.vecLen; j++) {
        this.weights[wtIdx] += this.workB[j] * this.lsh[wtIdx]; // adjust weights to give zero error
        wtIdx++;
      }
    }
  }

  recall(result, input) {
    copyVec(this.workA, input);
    zeroVec(result);
    let h = this.hash;
    let wtIdx = 0;
    for (let i = 0; i < this.density; i++) {
      rpVec(this.workA, h++); // random projection
      for (let j = 0; j < this.vecLen; j++) {
        const sign = this.workA[j] < 0 ? -1 : 1; // LSH bit
        result[j] += sign * this.weights[wtIdx++]; // weight each bit in the LSH and sum over density dimensions.
      }
    }
  }
  // Recall and store Locality Sensitive Hash so that it doesn't
  // need to be recalculated by the training method.
  recallLSH(result, input) {
    copyVec(this.workA, input);
    zeroVec(result);
    let h = this.hash;
    let wtIdx = 0;
    for (let i = 0; i < this.density; i++) {
      rpVec(this.workA, h++); // random projection
      for (let j = 0; j < this.vecLen; j++) {
        const sign = this.workA[j] < 0 ? -1 : 1; // LSH bit
        this.lsh[wtIdx] = sign; // Store to avoid recomputing LSH during training
        result[j] += sign * this.weights[wtIdx++]; // weight each bit in the LSH and sum over density dimensions.
      }
    }
  }

  clear() {
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = 0;
    }
  }
}

let vec = new Float32Array(4096);
let am = new AM(4096, 32, 0);
let img;
let side;
let trainingData = [];
let training = false;
let trainingCount = 0;

function preload() {
  img = loadImage('Lübeck.jpg');
  side = createImage(150, 400);
}

function setup() {
  createCanvas(800, 600);
}

function draw() {
  if (trainingData.length === 0) {
    background('grey');
  }
  image(side, 620, 0);
  image(img, 0, 0);
  if (training) {
    trainingCount++;
    fill(255);
    strokeWeight(10);
    textSize(30);
    text('Training Cycles:' + trainingCount, 5, 30);
    for (var i = 0; i < trainingData.length; i++) {
      am.train(trainingData[i], trainingData[i]);
    }
    return;
  }
  noFill();
  strokeWeight(1);
  if (mouseX < 600 - 33 && mouseY < 400 - 33) {
    square(mouseX - 1, mouseY - 1, 34);
  }
  if (((frameCount & 7) === 0) && (mouseX===pmouseX) && (mouseY===pmouseY)) {
    getData(vec, mouseX, mouseY);
    am.recall(vec, vec);
    setData(vec, 5, 405);
    copy(5, 405, 32, 32, 50, 405, 128, 128);
  }
}

function mouseClicked() {
  if (trainingData.length < 32) { // new training square   
    let x = trainingData.length;
    let y = int(x / 4);
    x %= 4;
    let sub = get(mouseX, mouseY, 32, 32);
    side.set(x * 40, y * 40 + 10, sub);
    let d = new Float32Array(4096);
    getData(d, mouseX, mouseY);
    trainingData.push(d);
  }
}

function keyPressed() {
  if (keyCode === 49) { // 1  train
    if (training) {
      training = false;
    } else {
      training = trainingData.length > 0;
      trainingCount = 0;
    }
  }
  if (keyCode === 48) { // 0 delete training squares
    trainingData = [];
    am.clear();
    side=createImage(150, 400);
  }
}

function getData(d, x, y) {
  let idx = 0;
  for (let px = 0; px < 32; px++) {
    for (let py = 0; py < 32; py++) {
      let c = get(x + px, y + py);
      d[idx++] = red(c) - 127.5;
      d[idx++] = green(c) - 127.5;
      d[idx++] = blue(c) - 127.5;
      d[idx++] = 0;
    }
  }
}

function setData(d, x, y) {
  let idx = 0;
  for (let px = 0; px < 32; px++) {
    for (let py = 0; py < 32; py++) {
      let r = constrain(d[idx++] + 127.5, 0, 255);
      let g = constrain(d[idx++] + 127.5, 0, 255);
      let b = constrain(d[idx++] + 127.5, 0, 255);
      idx++;
      set(x + px, y + py, color(r, g, b));
    }
  }
  updatePixels();
}


// Fast Walsh Hadamard Transform
function whtVec(vec) {
  let n = vec.length;
  let hs = 1;
  while (hs < n) {
    let i = 0;
    while (i < n) {
      const j = i + hs;
      while (i < j) {
        var a = vec[i];
        var b = vec[i + hs];
        vec[i] = a + b;
        vec[i + hs] = a - b;
        i += 1;
      }
      i += hs;
    }
    hs += hs;
  }
  scaleVec(vec, vec, 1.0 / Math.sqrt(n));
}

function signFlipVec(vec, hash) {
  for (let i = 0, n = vec.length; i < n; i++) {
    hash += 0x3C6EF35F;
    hash *= 0x19660D;
    hash &= 0xffffffff;
    if (((hash * 0x9E3779B9) & 0x80000000) === 0) {
      vec[i] = -vec[i];
    }
  }
}

// Fast random projection
function rpVec(vec, hash) {
  signFlipVec(vec, hash);
  whtVec(vec);
}

function scaleVec(rVec, xVec, sc) {
  for (let i = 0, n = rVec.length; i < n; i++) {
    rVec[i] = xVec[i] * sc;
  }
}

function multiplyVec(rVec, xVec, yVec) {
  for (let i = 0, n = rVec.length; i < n; i++) {
    rVec[i] = xVec[i] * yVec[i];
  }
}

function multiplyAddToVec(rVec, xVec, yVec) {
  for (let i = 0, n = rVec.length; i < n; i++) {
    rVec[i] += xVec[i] * yVec[i];
  }
}

// x-y
function subtractVec(rVec, xVec, yVec) {
  for (let i = 0, n = rVec.length; i < n; i++) {
    rVec[i] = xVec[i] - yVec[i];
  }
}

function addVec(rVec, xVec, yVec) {
  for (let i = 0, n = rVec.length; i < n; i++) {
    rVec[i] = xVec[i] + yVec[i];
  }
}

// converts each element of xVec to +1 or -1 according to its sign.
function signOfVec(rVec, xVec) {
  for (let i = 0, n = rVec.length; i < n; i++) {
    if (xVec[i] < 0.0) {
      rVec[i] = -1.0;
    } else {
      rVec[i] = 1.0;
    }
  }
}

function truncateVec(rVec, xVec, t) {
  for (let i = 0, n = rVec.length; i < n; i++) {
    let tt = Math.abs(xVec[i]) - t;
    if (tt < 0.0) {
      rVec[i] = 0.0;
      continue;
    }
    if (xVec[i] < 0.0) {
      rVec[i] = -tt;
    } else {
      rVec[i] = tt;
    }
  }
}

function sumSqVec(vec) {
  let sum = 0.0;
  for (let i = 0, n = vec.length; i < n; i++) {
    sum += vec[i] * vec[i];
  }
  return sum;
}

// Adjust variance/sd
function adjustVec(rVec, xVec, scale) {
  let MIN_SQ = 1e-20;
  let adj = scale / Math.sqrt((sumSq(xVec) / xVec.length) + MIN_SQ);
  scaleVec(rVec, xVec, adj);
}

function copyVec(rVec, xVec) {
  for (let i = 0, n = rVec.length; i < n; i++) {
    rVec[i] = xVec[i];
  }
}

function zeroVec(x) {
  for (let i = 0, n = x.length; i < n; i++) {
    x[i] = 0;
  }
}