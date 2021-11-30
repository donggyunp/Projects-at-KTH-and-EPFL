%create M^input_{x,k,n}
%The function then outputs the matrix MX of size (nlen-k+1)*nf × k*nf*d.
function MX = MakeMXMatrix(x_input, d, k, nf)

n_len = size(x_input,1)/d;
X_flat = x_input(:)';

MX = zeros( (n_len-k+1)*nf, k*nf*d);

for i = 1:(n_len-k+1)
    for j = 1:nf
        MX((i-1)*nf+j, (j-1)*d*k +1 : (j-1)*d*k +d*k  ) = X_flat(1,(i-1)*d +1:(i-1)*d+d*k);
    end
end


