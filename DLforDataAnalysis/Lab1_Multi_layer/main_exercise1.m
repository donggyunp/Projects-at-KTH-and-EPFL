

%configurations
K = 10;
lambda = 0;
GDparam=hyper;
GDparam.epoch = 30;
GDparam.eta = 0.001;
GDparam.size_batch = 100;

%load the data
[Xtrain,Ytrain,ytrain]=LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_1');
[Xvalid,Yvalid,yvalid]=LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_2');
[Xtest,Ytest,ytest]=LoadBatch('./Datasets/cifar-10-batches-mat/test_batch');

mean_X = mean(Xtrain, 2);
std_X = std(Xtrain, 0, 2);

Xtrain = Xtrain - repmat(mean_X, [1, size(Xtrain, 2)]);
Xtrain = Xtrain ./ repmat(std_X, [1, size(Xtrain, 2)]);
Xvalid = Xvalid - repmat(mean_X, [1, size(Xtrain, 2)]);
Xvalid = Xvalid ./ repmat(std_X, [1, size(Xtrain, 2)]);
Xtest = Xtest - repmat(mean_X, [1, size(Xtrain, 2)]);
Xtest = Xtest ./ repmat(std_X, [1, size(Xtrain, 2)]);

%initialize W and b
W = 0.01.*randn(K,3072);
b = 0.01.*randn(K,1);

%checking the gradient
%{
P_check = EvaluateClassifier(Xtrain(:,10), W, b);
[gW_me, gb_me] = ComputeGradient(Xtrain(:,10), Ytrain(:,10), P_check, W, lambda);
[gb_check,gW_check] =ComputeGradsNumSlow(Xtrain(:,10),Ytrain(:,10),W,b,lambda,1e-6);
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

J_train(1,1) = ComputeCost(Xtrain, Ytrain, W, b, lambda);
acc_train(1,1) = ComputeAccuracy(Xtrain, ytrain, W, b);
J_valid(1,1) = ComputeCost(Xvalid, Yvalid, W, b, lambda);
acc_valid(1,1) = ComputeAccuracy(Xvalid, yvalid, W, b);

for i = 1:GDparam.epoch
    %shuffle
    %{
    cols = size(Xtrain,2);
    P = randperm(cols);
    Xtrain = Xtrain(:,P);
    Ytrain = Ytrain(:,P);
    ytrain = ytrain(:,P);
    %}
    [W,b] = MiniBatchGD(Xtrain,Ytrain,GDparam,W,b,lambda);
    
    J_train(1,i+1) = ComputeCost(Xtrain, Ytrain, W, b, lambda);
    J_valid(1,i+1) = ComputeCost(Xvalid, Yvalid, W, b, lambda);
    acc_valid(1,i+1) = ComputeAccuracy(Xvalid, yvalid, W, b);
    acc_train(1,i+1) = ComputeAccuracy(Xtrain, ytrain, W, b);
end

J_test = ComputeCost(Xtest, Ytest, W, b, lambda)
acc_test = ComputeAccuracy(Xtest,ytest,W,b)

%plotting weights

figure(1);
for i=1:10
    im = reshape(W(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
    subplot(1,10,i);
    imagesc(s_im{i})
end


%plotting graphs
figure(2); %plot cost
subplot(1,2,1);
plot(J_train);
title('cross entropy loss on training')
xlabel('epoch') 
ylabel('loss')

subplot(1,2,2);
plot(J_valid);
title('cross entropy loss on validation')
xlabel('epoch') 
ylabel('loss')

figure(3); %plot accuracy
plot([0:1:GDparam.epoch],acc_train*100,'r',[0:1:GDparam.epoch],acc_valid*100,'b');
legend({'training accuracy','validation accuracy'},'Location','southeast');
xlabel('epoch') 
ylabel('accuracy(%)')