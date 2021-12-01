function result = MakeMFMatrix_improved(X,h,k)

n_len = size(X,1)/h;
X_flat = X(:)';
result = zeros(n_len-k+1, h*k);

for i = 1:(n_len-k+1)
        result(i,:) = X_flat(1,(i-1)*h +1:(i-1)*h+h*k);
end

