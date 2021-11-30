function J = ComputeCost(X ,Y, W, b, lambda)
%P = zeros(size(W,1),size(X,2));
P = EvaluateClassifier(X,W,b); 
J = 1/size(X,2)*sum(sum(Y.*(-log(P)))) + lambda*sum(sum(W.^2));
%J = 1/size(X,2)*sum(sum(-log(Y'*P))) + lambda*sum(sum(W.^2));
