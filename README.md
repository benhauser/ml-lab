# ml-lab

Machine Learning Laboratory / Experiments

## Machine Learning, what language should I use?

I'm trying to figure out the best programming language to use for Machine Learning. My three candidates are:

1. Octave (Matlab). This is the language chosen by Andrew Ng for his excellent [Machine Learning](https://www.coursera.org/learn/machine-learning) course at Stanford. Andrew said this was a carefully considered decision based on his experience that students learn more quickly in this high-level language.
1. Python. This seems to be the most popular choice for Machine Learning out in industry, let's try to find out why.
1. JavaScript. I considered JavaScript because the laguage is so ubiquitous and convenient. I've been doing a lot of JavaScript lately (haven't we all?) and I know I could deploy my Machine Learning programs in a browser directly if I went down this path.

## Linear Algebra

One of the mild surprises you experience when you get in to Machine Learning is that, under the hood, it's largely just Linear Algebra. Nothing fancy or difficult. Just good old Matrices and Vectors from high school mathematics.

I still remember with some fondness my Linear Algebra textbook from Maths II in my senior year. It was called "Matrices and Vectors" and it had a floppy, green cover with yellowing paper inside.

So my original question has now become: What's the best programming language for doing Linear Algebra?

## The Experiment

I implemented a typical Machine Learning problem in each language.

The Linear Algebra was done using [numpy](http://www.numpy.org/) for Python and [mathjs](http://mathjs.org/) for JavaScript.

Let's see how it turned out. Here are three key parts of the solution in each language.

### 1. Parsing Training Data

The training data is has been loaded into the variable `data`. This code separates the data into two column vectors and counts the number of training examples `m`.

Octave

```matlab
X = data(:, 1);
y = data(:, 2);
m = length(y);
```

Python

```python
X = data[:, 0:1]
y = data[:, 1:2]
m = len(y)
```
JavaScript

```javascript
var X = math.subset(data, math.index(math.range(0,m),0));
var y = math.subset(data, math.index(math.range(0,m),1));
var m = math.size(y)[0];
```

You can see that the Octave and Python code looks quite similar. The tricky part of the Python solution was to use slice indexing (`0:1` instead of `0`) to maintain rank-2 arrays. The JavaScript solution is very verbose in comparison.

### 2. Cost Function

A typical linear regression cost function in each language.

Octave

```matlab
function J = computeCost(X, y, theta)
    h = X * theta;
    err = h - y;
    J = 1 / (2 * m) * err' * err;
end
```

Python

```python
def computeCost(X, y, theta):
    h = np.dot(X, theta)
    err = h - y
    return 1.0 / (2.0 * m) * np.dot(err.T, err)
```

JavaScript

```javascript
function computeCost(X, y, theta) {
    var h = math.multiply(X, theta);
    var err = math.subtract(h, y);
    return 1 / (2 * m) * math.multiply(math.transpose(err), err);
}
```

The Octave solution is wonderfully concise and elegant.

The Python solution comes close. Unfortunately the asterisk cannot be overridden to perform matrix multiplication, we have to resort to a function call `dot()`. This pollutes things somewhat and is a bit of a drag.

Once again the JavaScript solution is pretty darn ugly. All our matrix operations require a function call: `multiply()`, `subtract()`, `transpose()`.

### 3. Gradient Descent

Octave

```matlab
function theta = gradientDescent(X, y, theta, alpha, num_iters)
    for iter = 1:num_iters
	   h = X * theta;
	   err = h - y;
	   theta_change = alpha / m * (X' * err);
	   theta = theta - theta_change;
    end
end
```

Python

```python
def gradientDescent(X, y, theta, alpha, num_iters):
    for i in range(0, num_iters):
        h = np.dot(X, theta)
        err = h - y
        theta_change = alpha / m * np.dot(X.T, err)
        theta = theta - theta_change
    return theta
```

JavaScript

```javascript
function gradientDescent(X, y, theta, alpha, num_iters) {
    for (var i = 0; i < num_iters; i++) {
        var h = math.multiply(X, theta);
        var err = math.subtract(h, y);
        var theta_change = math.multiply(alpha / m, math.multiply(math.transpose(X), err));
        theta = math.subtract(theta, theta_change);
    }
    return theta;
}
```

Very similar results to the cost function snippet. Octave is the most elegant. Python has the annoying `dot()` function call. And JavaScript is a hot mess.

## Conclusion

Octave is the simplest and cleanest language of these three for performing Linear Algebra. It is therefore my recommendation for learning, studying, and prototyping machine learning problems.

Python is close behind Octave in succintness. It has other things going for it however. It's a mainstream programming language with a huge userbase and library support. This makes it the go to choice for machine learning in industry.

JavaScript is a clunky choice for performing Linear Algebra. I cannot recommend it as a primary platform for Machine Learning work. This hasn't stopped motivated people from [going ahead](https://cs.stanford.edu/people/karpathy/convnetjs/) and [doing it anyway](https://deeplearnjs.org/) so YMMV.

So, in conclusion, if you are a researcher and/or interested in understanding and manipulating machine learning algorithms at a low level then use Octave.

If you are in industry and are applying machine learning algorthms at scale then use Python.

Try to avoid JavaScript if you can.
