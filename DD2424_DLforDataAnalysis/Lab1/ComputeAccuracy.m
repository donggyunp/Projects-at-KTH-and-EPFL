function acc = ComputeAccuracy(X, y, W, b)

P = zeros(size(W,1),size(X,2));
P = EvaluateClassifier(X, W, b);

argvalue = zeros(1,size(X,2));
argmax = zeros(1,size(X,2));
[argvalue, argmax] = max(P);
acc = 0;

for i = 1:size(X,2)
    if y(1,i) == argmax(1,i)
        acc = acc + 1;
    end
end
acc = acc/size(X,2);
