function [Wstar] = MiniBatchGD_SVM(X, Y, GDparams, W, lambda, margin)

n = size(X,2);

for j = 1 : n / GDparams.size_batch
  
            j_start = (j-1)*GDparams.size_batch + 1;
            j_end = j * GDparams.size_batch;
            %inds = j_start:j_end;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
            grad_W = ComputeGradient_SVM(Xbatch, Ybatch, W, lambda, margin);
            Wstar = W - GDparams.eta * grad_W;
            W = Wstar;
end

%Wstar=W;