# Machine Learning, what language should I use?

I'm trying to figure out the best programming language to use for Machine Learning. My three candidates are:

1. Octave (Matlab). This is the language chosen by Andrew Ng for his excellent [Machine Learning](https://www.coursera.org/learn/machine-learning) course at Stanford. Andrew has stated this was a carefully considered decision based on his experience that students learn more quickly in this high-level language.
1. Python. This seems to be the most popular choice for Machine Learning in industry.
1. JavaScript. I considered JavaScript because the language is so ubiquitous and convenient. I've been doing a lot of JavaScript lately (haven't we all?) and I knew I could show off my Machine Learning programs in a browser directly if I went down this path.

## Linear Algebra

One of the first surprises you experience when you dig in to (the current generation of) Machine Learning techniques is that, under the hood, they're largely just applied [Linear Algebra](https://en.wikipedia.org/wiki/Linear_algebra). Nothing fancy or difficult. Just good old Matrices and Vectors from high school mathematics.

I still remember with some fondness my Linear Algebra textbook from Maths II in my senior year. It was called _Matrices and Vectors_ and it had a floppy, green cover with yellowing paper inside.

So my titular question has now become: _Linear Algebra, what language should I use?_

## The Experiment

I implemented a typical Machine Learning problem in each language.

The Linear Algebra parts were done using [numpy](http://www.numpy.org/) for Python and [mathjs](http://mathjs.org/) for JavaScript.

Let's see how it turned out. We'll compare three key parts of the solution in each language.

### 1. Processing Training Data

Assume the training data has been loaded into the variable `data`. This code separates the data into two column vectors and counts the number of training examples `m`.

#### Octave

```matlab
X = data(:, 1);
y = data(:, 2);
m = length(y);
```

#### Python

```python
X = data[:, 0:1]
y = data[:, 1:2]
m = len(y)
```

#### JavaScript

```javascript
var X = math.subset(data, math.index(math.range(0, m), 0));
var y = math.subset(data, math.index(math.range(0, m), 1));
var m = math.size(y)[0];
```

You can see that Octave and Python look quite similar. The tricky part of the Python solution was to use slice indexing (`0:1` instead of `0`) to maintain rank-2 arrays. The JavaScript solution is very verbose in comparison.

### 2. Cost Function

Now let's examine a typical linear regression cost function in each language.

#### Octave

```matlab
function J = computeCost(X, y, theta)
    h = X * theta;
    err = h - y;
    J = 1 / (2 * m) * err' * err;
end
```

#### Python

```python
def computeCost(X, y, theta):
    h = np.dot(X, theta)
    err = h - y
    return 1.0 / (2.0 * m) * np.dot(err.T, err)
```

#### JavaScript

```javascript
function computeCost(X, y, theta) {
    var h = math.multiply(X, theta);
    var err = math.subtract(h, y);
    return 1 / (2 * m) * math.multiply(math.transpose(err), err);
}
```

The Octave solution is wonderfully concise and elegant.

The Python solution comes close. We use numpy's `array` datatype as opposed to the `matrix` datatype ([as recommended](http://scipy.github.io/old-wiki/pages/NumPy_for_Matlab_Users#A.27array.27_or_.27matrix.27.3F_Which_should_I_use.3F)). The only downside of this that we must resort to the function call `dot()` to perform matrix multiplication. This pollutes things somewhat and is a bit of a drag.

Once again the JavaScript solution is really quite ugly. _Every_ matrix operation requires a function call: `multiply()`, `subtract()`, `transpose()`.

### 3. Gradient Descent

#### Octave

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

#### Python

```python
def gradientDescent(X, y, theta, alpha, num_iters):
    for i in range(0, num_iters):
        h = np.dot(X, theta)
        err = h - y
        theta_change = alpha / m * np.dot(X.T, err)
        theta = theta - theta_change
    return theta
```

#### JavaScript

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

Very similar results to the _Cost Function_. Octave is the most elegant. Python is ok apart from that annoying `dot()` function call. And JavaScript is a hot mess.

## Conclusion

Octave has the simplest and cleanest syntax for performing Linear Algebra. It's a great choice for learning, studying, and prototyping Machine Learning problems.

Python is close behind Octave in succintness. It has other things going for it however. It's a mainstream programming language with a huge userbase and massive library support. This makes it the go to choice for Machine Learning in industry.

JavaScript is a clunky choice for performing Linear Algebra / Machine Learning. This hasn't stopped motivated people from [going ahead](https://cs.stanford.edu/people/karpathy/convnetjs/) and [doing it anyway](https://deeplearnjs.org/) so your mileage may vary.

In conclusion, if you are a researcher and/or interested in understanding and manipulating Machine Learning algorithms at a low level then Octave is a fine choice.

If you are in industry and are applying Machine Learning algorthms at scale then Python might be the best option.

It's probably best to avoid JavaScript if you can.
