    function J = ComputeCost(X ,Y, W1, b1, W2, b2, lambda)
%P = zeros(size(W,1),size(X,2));
P = EvaluateClassifier(X,W1,b1,W2,b2); 
J = 1/size(X,2)*sum(sum(Y.*(-log(P)))) + lambda*(sum(sum(W2.^2))+sum(sum(W1.^2)));
%J = 1/size(X,2)*sum(sum(-log(Y'*P))) + lambda*sum(sum(W.^2));
