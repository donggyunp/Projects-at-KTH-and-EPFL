%create matrix M_{Fx,n_len}
function MF = MakeMFMatrix(F, n_len)

%The function then outputs the matrix MF of size (nlen-k+1)*nf × nlen*dd where
[dd, k, nf] = size(F);

MF = zeros( (n_len-k+1)*nf , n_len*dd );
F_flat = F(:)';

for i = 1:(n_len-k+1)
    for j = 1:nf
        MF((i-1)*nf +j, (i-1)*dd+1 : (i-1)*dd + size(F_flat,2)/nf)...
        = F_flat(1,(j-1)*dd*k+1 : j*dd*k);
    end
end


