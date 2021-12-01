function J = ComputeCost_SVM(X, Y, W, b, lambda, margin)

%P = EvaluateClassifier(X,W,b); 
%for single datapoint
P = EvaluateClassifier(X, W, b);
J=0;
Y=int32(Y);
[argvalue, argmax] = max(Y);
for i = 1:size(X,2)
    for j = 1:size(W,1)
        if Y(j,i) ~= 1        
            J = J + max(P(j,i)-P(argmax(1,i),i)+margin,0);
        end
    end
end
J = J/size(X,2);
%J = J + lambda*sum(sum(W.^2));
%J = 1/size(X,2)*sum(sum(-log(Y'*P))) + lambda*sum(sum(W.^2));
