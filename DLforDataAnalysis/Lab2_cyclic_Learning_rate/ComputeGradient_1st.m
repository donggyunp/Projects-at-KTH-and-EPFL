function [grad_W, grad_b] = ComputeGradient_1st(X, Y, H, P, W1, W2, lambda)

%G = zeros(size(Y));
G = -(Y - P);
G = W2'*G; %W2 = 10 X 50, so G becomes 50 X batch_size
ind = zeros(size(H)); % H = 50 X batch_size
for i = 1:size(H,1)
    for j = 1:size(H,2)
        if H(i,j) > 0
            ind(i,j) = 1;
        end
    end
end
G = G.*ind;
%grad_L_W = zeros(size(G,1),size(X,1));
%grad_L_b = zeros(size(W,1),1);

grad_L_W = 1/size(X,2) * G * X'; % [50 X batch_size] X [batch_size X 3072]
grad_L_b = 1/size(X,2) * G * ones(size(X,2),1);

%grad_W = zeros(size(G,1),size(X,1));
grad_W = grad_L_W + 2*lambda.*W1; %It should be same form with W1 = 50 X 3072 
%grad_b = zeros(size(W,1),1);
grad_b = grad_L_b;