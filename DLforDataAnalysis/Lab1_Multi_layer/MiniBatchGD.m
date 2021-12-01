function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)
%이 함수안에서 배치만들어서 다 계산하고 스타값리턴
n = size(X,2);
%Xbatch = zeros(size(X,1),size(X,2)/GDparams.size_batch);
%Ybatch = zeros(size(Y,1),size(Y,2)/GDparams.size_batch);
%grad_W = zeros(size(W));
%grad_b = zeros(size(b));

for j = 1 : n / GDparams.size_batch
  
            j_start = (j-1)*GDparams.size_batch + 1;
            j_end = j * GDparams.size_batch;
            %inds = j_start:j_end;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
            P = EvaluateClassifier(Xbatch, W, b);
            [grad_W, grad_b] = ComputeGradient(Xbatch, Ybatch, P, W, lambda);

            W = W - GDparams.eta * grad_W;
            b = b - GDparams.eta * grad_b;

end

Wstar=W;
bstar=b;