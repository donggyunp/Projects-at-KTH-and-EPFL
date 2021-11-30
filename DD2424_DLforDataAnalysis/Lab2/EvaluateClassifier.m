function P = EvaluateClassifier(X, W1, b1, W2, b2)

%s = zeros(size(W,1),size(X,2));
%P = zeros(1,size(X,2));

s= W1*X + b1;
H= max(s,0);
P= softmax(W2*H + b2);

