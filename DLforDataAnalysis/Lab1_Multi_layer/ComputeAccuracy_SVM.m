function acc = ComputeAccuracy_SVM(X, Y, W, margin)


eval = W*X;
[argvalue, argmax] = max(Y);
acc = 0;
for i = 1:size(X,2)
    classified = 0;
    for j = 1:size(W,1)
        if eval(j,i)-eval(argmax(1,i),i)+margin < 0
            classified = classified + 1;
        end
    end
    if classified == 9
        acc = acc + 1;
    end
end
acc = acc/size(X,2);
    