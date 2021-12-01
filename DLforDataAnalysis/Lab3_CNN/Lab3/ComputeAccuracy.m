function acc = ComputeAccuracy(X, Y, ConvNet,n_len,n_len1)

MF1 = MakeMFMatrix(ConvNet.F1,n_len);
MF2 = MakeMFMatrix(ConvNet.F2,n_len1);
X_1 = max(0, MF1*X);
X_2 = max(0, MF2*X_1);
S = ConvNet.W * X_2;
P = softmax(S);

%argvalue = zeros(1,size(X,2));
%argmax = zeros(1,size(X,2));
[argvalue, argmax] = max(P);
[argvalue2, argmax2] = max(Y);
acc = 0;

for i = 1:size(X,2)
    if argmax2(1,i) == argmax(1,i)
        acc = acc + 1;
    end
end
acc = acc/size(X,2)*100;

%sizeP = size(P)
%sizeY = size(Y)
