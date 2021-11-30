load('X');

Convmain = ConvNet();
Convmain.eta = 0.001;
Convmain.rho = 0.9;
epoch = 1500;
batch = 100;

%the number of filters applied at layer 1: n1
%the width of the filters applied at layer 1: k1
%(Remember the filter applied will have size d × k1.)
%the number of filters applied at layer 2: n2
%the width of the filters applied at layer 2: k2
%(Remember the filter applied will have size n1 × k2.)

%congifuration
k1=5; n1=20;
k2=3; n2=20;
K=18;

Convmain.Initialize(d,k1,n1,k2,n2,K,n_len);

n_len1 = n_len-k1+1;
n_len2 = n_len1 -k2 +1;
fsize=n2*n_len2;

[d, k1, nf] = size(Convmain.F1);

y=ys';
n_c = zeros(K,1);
Y = zeros(K,size(ys,1));
for i = 1:size(Y,2)
    Y(ys(i,1),i)=1;
end
n=size(X,2);
temp = randperm(n);
X = X(:,temp);
Y = Y(:,temp);
y = y(:,temp);
X_valid = X(:,1:4000);
Y_valid = Y(:,1:4000);
X_train = X(:,4001:17421);
Y_train = Y(:,4001:17421);
y_train = y(:,4001:17421);
n_t=size(X_train,2);
for i = 1:n_t
    n_c(y_train(1,i),1) = n_c(y_train(1,i),1) + 1;
end
min_class = min(n_c)

train_loss = [];
train_acc = [];
valid_loss = [];
valid_acc = [];
Convmain.moment_W = 0;
Convmain.moment_F1 = 0;
Convmain.moment_F2 = 0;

%profile on

for i = 1:epoch
    temp = randperm(n_t);
    X_train = X_train(:,temp);
    Y_train = Y_train(:,temp);
    
    %%%%balanced
    [X_bal,Y_bal] = MakeCompen(X_train,Y_train,K,min_class);
    n_bal = size(X_bal,2);
    
    %%%%unbalanced
    %n_bal = size(X_train,2);
    
    for a = 1:n_bal/batch
        %%%%%%balanced
        X_batch = X_bal(:,(a-1)*batch+1:a*batch);
        Y_batch = Y_bal(:,(a-1)*batch+1:a*batch);
        
        %%%%%%unbalanced
        %X_batch = X_train(:,(a-1)*batch+1:a*batch);
        %Y_batch = Y_train(:,(a-1)*batch+1:a*batch);
        
        Convmain.forward(X_batch,n_len);
        [train_loss1,train_acc1] = Convmain.ComputeLossandAcc(X_batch,Y_batch);
        [valid_loss1,valid_acc1] = Convmain.Compute_valid(X_valid,Y_valid,n_len);
        Convmain.backprop(X_batch,Y_batch,n_len);
        MF1 = MakeMFMatrix(Convmain.F1, n_len);
        train_loss = cat(2,train_loss,train_loss1);
        train_acc = cat(2,train_acc,train_acc1);
        valid_loss = cat(2,valid_loss,valid_loss1);
        valid_acc = cat(2,valid_acc,valid_acc1);
        %the beginning of for loop
        %Gs = NumericalGradient(X_batch, Y_batch, Convmain, 1e-6);
        
    end
end
%profile viewer






    