function acc = ComputeAccuracy_SVM2(X, y, Y, W, margin)


eval = W*X;
[argvalue, argmax] = max(Y);
acc = 0;

for i = 1:size(X,2)
    if y(1,i) == argmax(1,i)
        acc = acc + 1;
    end
end
acc = acc/size(X,2);

    