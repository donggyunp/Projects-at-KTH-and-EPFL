function [Y] = synthesize(p)
[K,n] = size(p);
cp = cumsum(p);
a = rand;
ixs = find(cp-a >0);
ii = ixs(1);
%You should store each index you sample for 1=< t =<n and let your function output
%the matrix Y (size K x n) where Y is the one-hot encoding of each sampled
%character. Given Y you can then use the map container ind to char to
%convert it to a sequence of characters and view what text your RNN has
%generated.
Y = zeros(K,n);
y = zeros(1,n);
for t = 1:n
    a = rand;
    ixs = find(cp-a >0);
    ii = ixs(1);
    y(1,t) = ii;
    Y(y(1,t),t) = 1;
end