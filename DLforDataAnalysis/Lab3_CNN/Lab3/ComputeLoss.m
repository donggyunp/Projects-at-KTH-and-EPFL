function loss = ComputeLoss(Convmain, MFs)

%where X batch is matrix of size nlen*d × n
%that contains all the vectorised input data
%(each column corresponds to one vectorized datapoint),
%Y batch is the one-hot encoding of the labels of each example,
%MFs is a cell array containing the MF matrix for each layer
%and W is the 2D weight matrix for the last fully connected layer

%cross entropy
n = size(Convmain.X_batch,2);
loss = 0;
for i = 1:n
    Convmain.X_batch_1 = max(MFs{1}*Convmain.X_batch, 0);
    %Each x(1) has size nlen1 ×1 where nlen1 = nlen −k1 +1
    %which X(1) has size n1×nlen1.
    Convmain.X_batch_2 = max(MFs{2} * Convmain.X_batch_1 ,0);
    Convmain.S_batch = Convmain.W * Convmain.X_batch_2;
    Convmain.P_batch = softmax(Convmain.S_batch)
    loss=sum(sum(-log(Convmain.Y_batch' * Convmain.P_batch)));
end
