var fs = require('fs');
var math = require('mathjs');

// TODO a library might make this csv parse easier
var raw = fs.readFileSync('ex1data1.txt', 'utf8');
var values = raw.split(/\n|,/).map(Number); // parse the csv into one flat array of Numbers
var m = values.length / 2;
var data = math.reshape(values, [m, 2]); // reshape to (97,2)
var X = math.subset(data, math.index(math.range(0,m),0)); // data(:, 1);
var y = math.subset(data, math.index(math.range(0,m),1)); // data(:, 2);
var m = math.size(y)[0];

var X = math.concat(math.ones(m,1), X); //  Add a column of ones to x
var theta = math.zeros(2, 1); // initialize fitting parameters
var iterations = 1500;
var alpha = 0.01;

console.log('\nTesting the cost function ...');
// compute and display initial cost
var J = computeCost(X, y, theta);
console.log('With theta = [0 ; 0]\nCost computed = ' + J);
console.log('Expected cost value (approx) 32.07');

function computeCost(X, y, theta) {
    var h = math.multiply(X, theta); // (97,2) * (2,1) = (97,1)
    var err = math.subtract(h, y); // (97,1) - (97,1) = (97,1)
    return 1 / (2 * m) * math.multiply(math.transpose(err), err); // (1,97) * (97,1) = scalar
}

// further testing of the cost function
J = computeCost(X, y, [[-1],[2]]);
console.log('\nWith theta = [-1 ; 2]\nCost computed = ' + J);
console.log('Expected cost value (approx) 54.24');

function gradientDescent(X, y, theta, alpha, num_iters) {
    for (var i = 0; i < num_iters; i++) {
        var h = math.multiply(X, theta); // (97,2) * (2,1) = (97,1)
        var err = math.subtract(h, y); // (97,1) - (97,1) = (97,1)
        var theta_change = math.multiply(alpha / m, math.multiply(math.transpose(X), err)); // (2,97) * (97,1) = (2,1)
        theta = math.subtract(theta, theta_change); // (2,1) - (2,1) = (2,1)
    }
    return theta;
}

console.log('\nRunning Gradient Descent ...\n')
// run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

// print theta to screen
console.log('Theta found by gradient descent:');
console.log(theta);
console.log('Expected theta values (approx)');
console.log(' -3.6303\n  1.1664\n\n');

// Predict values for population sizes of 35,000 and 70,000
var predict1 = math.multiply([1, 3.5], theta);
console.log('For population = 35,000, we predict a profit of ' + math.multiply(predict1,10000));
var predict2 = math.multiply([1, 7], theta);
console.log('For population = 70,000, we predict a profit of ' + math.multiply(predict2,10000));

//var X = [];
//var y = [];
//for (var i = 0; i < lines.length; i++) {
//    var line = lines[i].split(/,/);
//    X.push(line[0]);
//    y.push(line[1]);
//}
//console.log(lines.length);
//console.log(X);
//console.log(X.length);
//var data = np.genfromtxt('ex1data1.txt', delimiter=',')
//X = data[:, 0]
//y = data[:, 1]
//m = len(y); # number of training examples

/*
// training data (AND)
var train = [
    [0,0,0],
    [0,1,0],
    [1,0,0],
    [1,1,1],
];

var θ = [0, 0, 0];
var m = train.length;

console.log('m ' + m);
console.log('θ ' + θ);
//weights = randomizeWeights(weights);
//weights = [-30, 20, 20];
//weights = [-3, 2, 2];
//console.log('w ' + weights);

//h = sigmoid(X * theta);										% dimensions (100,3) * (3,1) = (100,1)
//J = (1 / m) * ((-y' * log(h)) - ((1 - y)' * log(1 - h)));	% dimensions (1,100) * (100,1) - (1,100) * (100,1) = scalar - scalar
//errors = h - y;					% dimensions (100,1) - (100,1) = (100,1)
//grad = 1 / m * (X' * errors);	% dimensions (3,100) * (100,1) = (3,1)

function cost(target, actual) {
    return (Math.log(actual) * -target) - (Math.log(1 - actual) * (1 - target));
}


// feedforward
for (var i = 0; i < train.length; i++) {
    var t = train[i];
    var a = feedforward(t);
    var l = cost(t[2], a);
    console.log('l ' + l);
}

// score
score = 0;
for (var i = 0; i < train.length; i++) {
    var t = train[i];
    var predict = (feedforward(t) > 0.5) ? 1 : 0;
    var actual = t[2];
    if (predict == actual) {
        score++;
    }
    console.log('predict ' + predict + ' actual ' + actual);
}
console.log('training score ' + score + '/' + train.length + ' = ' + (score/train.length * 100) + "%");

function feedforward(t) {
    var x = [1, t[0], t[1]];
    console.log('x ' + x);
    var z = multiply(x, θ);
    console.log('z ' + z);
    var a = sigmoid(z);    
    console.log('a ' + a);
    return a;
}
            
function sigmoid(z) {
    return 1.0 / (1.0 + Math.exp(-z));
}

function sigmoidGradient(z) {
    return sigmoid(z) * (1 - sigmoid(z));
}

// vector multiplication
function multiply(a, b) {
    var sum = 0;
    for (var i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

function randomizeWeights(weights) {
    return weights.map(x => Math.random());
}

*/