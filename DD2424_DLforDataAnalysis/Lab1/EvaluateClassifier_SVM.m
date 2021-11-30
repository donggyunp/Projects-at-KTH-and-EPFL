function P = EvaluateClassifier_SVM(X, W)

s = zeros(size(W,1),size(X,2));
P = zeros(1,size(X,2));

s= W*X;
P = softmax(s);
