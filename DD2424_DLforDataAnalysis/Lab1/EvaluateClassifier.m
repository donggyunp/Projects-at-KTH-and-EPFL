function P = EvaluateClassifier(X, W, b)

s = zeros(size(W,1),size(X,2));
P = zeros(1,size(X,2));

s= W*X + b;
P = softmax(s);

