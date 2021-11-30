function [grad_b, grad_W] = ComputeGradsNum(X, Y, W1, b1,W2,b2, lambda, h)

no = size(W1, 1);
d = size(X, 1);

grad_W = zeros(size(W1));
grad_b = zeros(no, 1);

c = ComputeCost(X, Y, W1, b1,W2,b2, lambda);

for i=1:length(b1)
    b_try = b1;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W1, b_try,W2,b2, lambda);
    grad_b(i) = (c2-c) / h;
end

for i=1:numel(W)   
    
    W_try = W1;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b1, W2,b2, lambda);
    
    grad_W(i) = (c2-c) / h;
end