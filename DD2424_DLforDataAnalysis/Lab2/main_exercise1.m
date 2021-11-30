

%configurations
m = 60;
K = 10;
l_min=-4;
l_max=-3;
l = l_min + (l_max - l_min)*rand(1, 1);
lambda = 10^l;
lambda = 0.0023;
GDparam = hypers;
GDparam.epoch = 100;
GDparam.eta_min = 0.00001;
GDparam.eta_max = 0.1;
GDparam.eta_s = 4900;
GDparam.size_batch = 100;
GDparam.eta_t=0;
GDparam.count = 0;
GDparam.l = -1;
%load the data
[Xtrain1,Ytrain1,ytrain1]=LoadBatch('Datasets/cifar-10-batches-mat/data_batch_1');
[Xtrain2,Ytrain2,ytrain2]=LoadBatch('Datasets/cifar-10-batches-mat/data_batch_2');
[Xtrain3,Ytrain3,ytrain3]=LoadBatch('Datasets/cifar-10-batches-mat/data_batch_3');
[Xtrain4,Ytrain4,ytrain4]=LoadBatch('Datasets/cifar-10-batches-mat/data_batch_4');
[Xtrain5,Ytrain5,ytrain5]=LoadBatch('Datasets/cifar-10-batches-mat/data_batch_5');

Xtrain = cat(2,Xtrain1,Xtrain2);
Xtrain = cat(2,Xtrain,Xtrain3);
Xtrain = cat(2,Xtrain,Xtrain4);
Xtrain = cat(2,Xtrain,Xtrain5(:,1:9000));
%temp=0.3+0.5*rand(3072,size(Xtrain,2));
%Xtrain = cat(2,Xtrain,Xtrain.*temp);

Ytrain = cat(2,Ytrain1,Ytrain2);
Ytrain = cat(2,Ytrain,Ytrain3);
Ytrain = cat(2,Ytrain,Ytrain4);
Ytrain = cat(2,Ytrain,Ytrain5(:,1:9000));
%Ytrain = cat(2,Ytrain,Ytrain);

ytrain = cat(2,ytrain1,ytrain2);
ytrain = cat(2,ytrain,ytrain3);
ytrain = cat(2,ytrain,ytrain4);
ytrain = cat(2,ytrain,ytrain5(:,1:9000));
%ytrain = cat(2,ytrain,ytrain);
Xtrain_f = zeros(3072,1);
Ytrain_f = zeros(10,1);
ytrain_f = zeros(1,1);
k=0;
%{
for ind=1:49000   
    if ytrain(1,ind)==0
        Xtrain_f=cat(2,Xtrain_f,flip(Xtrain(:,ind)));
        ytrain_f=cat(2,ytrain_f,[0]);
        Ytrain_f=cat(2,Ytrain_f,[1;0;0;0;0;0;0;0;0;0]);
        k=k+1;
        %Xtrain(2049:3072,ind)=flip(Xtrain(2049:3072,ind));
    end
    if ytrain(1,ind)==1
        Xtrain_f=cat(2,Xtrain_f,flip(Xtrain(:,ind)));
        ytrain_f=cat(2,ytrain_f,[1]);
        Ytrain_f=cat(2,Ytrain_f,[0;1;0;0;0;0;0;0;0;0]);
        k=k+1;
    end
    if ytrain(1,ind)==8
        Xtrain_f=cat(2,Xtrain_f,flip(Xtrain(:,ind)));
        ytrain_f=cat(2,ytrain_f,[8]);
        Ytrain_f=cat(2,Ytrain_f,[0;0;0;0;0;0;0;0;1;0]);
        k=k+1;
        %Xtrain(1025:2048,ind)=flip(Xtrain(1025:2048,ind));
        %Xtrain(2049:3072,ind)=flip(Xtrain(2049:3072,ind));
    end
end

Xtrain_f(:,1) = [];
Ytrain_f(:,1) = [];
ytrain_f(:,1) = [];
Xtrain=cat(2,Xtrain,Xtrain_f(:,1:9000));
%Xtrain=cat(2,Xtrain,Xtrain_f(:,1:7));
ytrain=cat(2,ytrain,ytrain_f(:,1:9000));
%ytrain=cat(2,ytrain,ytrain_f(:,1:7));
Ytrain=cat(2,Ytrain,Ytrain_f(:,1:9000));
%Ytrain=cat(2,Ytrain,Ytrain_f(:,1:7));
%}
Xvalid = Xtrain5(:,9001:10000);
Yvalid = Ytrain5(:,9001:10000);
yvalid = ytrain5(:,9001:10000);
[Xtest,Ytest,ytest]=LoadBatch('Datasets/cifar-10-batches-mat/test_batch');

%cost = zeros(1,GDparam.epoch * size(Xtrain,2) / GDparam.size_batch);
%acc = zeros(1,GDparam.epoch * size(Xtrain,2) / GDparam.size_batch);
%update_cost = double.empty(size(GDparam.epoch * size(Xtrain,2) / GDparam.size_batch));

mean_X = mean(Xtrain, 2);
std_X = std(Xtrain, 0, 2);

Xtrain = Xtrain - repmat(mean_X, [1, size(Xtrain, 2)]);
Xtrain = Xtrain ./ repmat(std_X, [1, size(Xtrain, 2)]);
Xvalid = Xvalid - repmat(mean_X, [1, size(Xvalid, 2)]);
Xvalid = Xvalid ./ repmat(std_X, [1, size(Xvalid, 2)]);
Xtest = Xtest - repmat(mean_X, [1, size(Xtest, 2)]);
Xtest = Xtest ./ repmat(std_X, [1, size(Xtest, 2)]);


%Checking if the gradient is correct; overfitting
%Xtrain = Xtrain(:,100);
%Xvalid = Xvalid(:,100);
%Xtrain = Xtrain(:,100);

%initialize W and b
W1 = 1/sqrt(3072).*randn(m,3072);
b1 = 0.*randn(m,1);
W2 = 1/sqrt(m).*randn(K,m);
b2 = 0.*randn(K,1);


%checking the gradient
%{
P_check = EvaluateClassifier(Xtrain(1:10,1), W1(1:10,1:10), b1(1:10,1), W2(:,1:10), b2(1:10,1));
[grad_b1, grad_W1] = ComputeGradsNum(Xtrain(1:20,1), Ytrain(:,1), W1(:,1:20), b1, lambda, 1e-5)
[grad_b2, grad_W2] = ComputeGradsNum(Xtrain(1:20,1), Ytrain(:,1), W2(:,1:20), b2, lambda, 1e-5)
[gW2_me, gb2_me] = ComputeGradient_2nd(H(:,10), Ytrain(:,1), P_check, W2, lambda);
[gW21_me, gb1_me] = ComputeGradient_1st(Xtrain(1:20,1), Ytrain(:,1), H(:,1), P_check, W1,W2, lambda);
if max(abs(gW_me - gW_check)) >  1e-6 | max(abs(gb_me - gb_check)) > 1e-6
    fprintf('wrong gradient!');
else
    fprintf('correct gradient!')
end
%}
%{
P_check = EvaluateClassifier(Xtrain(:,10), W1, b1);
[gW_me, gb_me] = ComputeGradient(Xtrain(:,10), Ytrain(:,10), P_check, W1, lambda);
[gb_check,gW_check] =ComputeGradsNumSlow(Xtrain(:,10),Ytrain(:,10),W1,b1,lambda,1e-6);
if max(abs(gW_me - gW_check)) >  1e-6 | max(abs(gb_me - gb_check)) > 1e-6
    fprintf('wrong gradient!');
else
    fprintf('correct gradient!')
end
%}
%start training
J_train = zeros(1,GDparam.epoch+1);
J_valid = zeros(1,GDparam.epoch+1);

acc_train = zeros(1,GDparam.epoch+1);
acc_valid = zeros(1,GDparam.epoch+1);

J_train(1,1) = ComputeCost(Xtrain, Ytrain, W1, b1, W2, b2, lambda);
acc_train(1,1) = ComputeAccuracy(Xtrain, ytrain, W1, b1, W2, b2);
J_valid(1,1) = ComputeCost(Xvalid, Yvalid, W1, b1, W2, b2, lambda);
acc_valid(1,1) = ComputeAccuracy(Xvalid, yvalid, W1, b1, W2, b2);

track_eta = zeros(1,GDparam.epoch * size(Xtrain,2) / GDparam.size_batch);
loss_update = zeros(1,GDparam.epoch * size(Xtrain,2) / GDparam.size_batch);
cost_t_update = zeros(1,GDparam.epoch * size(Xtrain,2) / GDparam.size_batch);
cost_v_update = zeros(1,GDparam.epoch * size(Xtrain,2) / GDparam.size_batch);
acc_t_update = zeros(1,GDparam.epoch * size(Xtrain,2) / GDparam.size_batch);
acc_v_update = zeros(1,GDparam.epoch * size(Xtrain,2) / GDparam.size_batch);
cnt_in = 0;
l_in = 0;
for i = 1:GDparam.epoch
        
    %shuffle
    cols = size(Xtrain,2);
    P = randperm(cols);
    Xtrain = Xtrain(:,P);
    Ytrain = Ytrain(:,P);
    ytrain = ytrain(:,P);
    [W1star, b1star, W2star, b2star, cnt_out,l_out, cost_t, acc_t, cost_v, acc_v, track]...
    = MiniBatchGD(cost_t_update,cost_v_update, acc_t_update, acc_v_update, Xtrain, ytrain, Ytrain, Xvalid, yvalid, Yvalid, GDparam, cnt_in, l_in, track_eta, W1,b1,W2,b2,lambda);
    %update record
    cnt_in = cnt_in + size(Xtrain,2) / GDparam.size_batch;
    l_in = l_out;
    cost_t_update = cost_t;
    cost_v_update = cost_v;
    acc_t_update = acc_t;
    acc_v_update = acc_v;
    track_eta = track;
    W1 = W1star;
    W2 = W2star;
    %{
    for a = 1:size(W1,1)
        for b = 1:size(W1,2)
            if abs(W1(a,b))>0.2
                W1(a,b)= 0.2;
            end
        end
    end
    for a = 1:size(W2,1)
        for b = 1:size(W2,2)
            if abs(W2(a,b))>0.2
                W2(a,b)= 0.2;
            end
        end
    end
    %}
    b1 = b1star;
    b2 = b2star;
    J_train(1,i+1)...
    = ComputeCost(Xtrain, Ytrain, W1star, b1star, W2star, b2star, lambda);
    J_valid(1,i+1)...
    = ComputeCost(Xvalid, Yvalid, W1star, b1star, W2star, b2star, lambda);
    acc_valid(1,i+1)...
    = ComputeAccuracy(Xvalid,yvalid,W1star,b1star,W2star,b2star);
    acc_train(1,i+1)...
    = ComputeAccuracy(Xtrain, ytrain, W1star, b1star, W2star, b2star);
    
    GDparam.eta_max=GDparam.eta_max*0.99
end

J_test = ComputeCost(Xtest, Ytest, W1star, b1star, W2star,b2star, lambda);
acc_test = ComputeAccuracy(Xtest,ytest,W1star,b1star,W2star,b2star);

%plotting learning rate
figure(1);
plot([1:1:cnt_out],track);


figure(2); %plot accuracy
plot([1:1:cnt_out],acc_t*100,'r',[1:1:cnt_out],acc_v*100,'b');
legend({'training accuracy','validation accuracy'},'Location','southeast');
xlabel('update') 
ylabel('accuracy(%)')

figure(3); %plot cost
plot([1:1:cnt_out],cost_t,'r',[1:1:cnt_out],cost_v,'b');
legend({'training loss','validation loss'},'Location','southwest');
xlabel('update') 
ylabel('Loss')

