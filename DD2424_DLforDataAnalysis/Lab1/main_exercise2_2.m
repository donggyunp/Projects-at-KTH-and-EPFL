

%configurations
K = 10;
lambda = 0.01;
GDparam=hyper_SVM;
GDparam.epoch = 40;
GDparam.eta = 0.1;
GDparam.margin = 0.5;
GDparam.size_batch = 100;

%load the data
[Xtrain,Ytrain,ytrain]=LoadBatch('Datasets/cifar-10-batches-mat/data_batch_1');
[Xvalid,Yvalid,yvalid]=LoadBatch('Datasets/cifar-10-batches-mat/data_batch_2');
[Xtest,Ytest,ytest]=LoadBatch('Datasets/cifar-10-batches-mat/test_batch');

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

%start training
J_train = zeros(1,GDparam.epoch+1);
J_valid = zeros(1,GDparam.epoch+1);

acc_train = zeros(1,GDparam.epoch+1);
acc_valid = zeros(1,GDparam.epoch+1);
J = ComputeCost_SVM(Xtrain,Ytrain,W,b,0,GDparam.margin);

J_train(1,1) = ComputeCost_SVM(Xtrain, Ytrain, W, b, lambda,GDparam.margin);
acc_train(1,1) = ComputeAccuracy_SVM(Xtrain, Ytrain, W, GDparam.margin);
J_valid(1,1) = ComputeCost_SVM(Xvalid, Yvalid, W, b, lambda,GDparam.margin);
acc_valid(1,1) = ComputeAccuracy(Xvalid, Yvalid, W, GDparam.margin);

for i = 1:GDparam.epoch
    %shuffle
    cols = size(Xtrain,2);
    temp = randperm(cols);
    Xtrain = Xtrain(:,temp);
    Ytrain = Ytrain(:,temp);
    ytrain = ytrain(:,temp);
    W = MiniBatchGD_SVM(Xtrain,Ytrain,GDparam,W,lambda,GDparam.margin);
    
    J_train(1,i+1) = ComputeCost_SVM(Xtrain, Ytrain, W, b, lambda,GDparam.margin);
    acc_valid(1,i+1) = ComputeAccuracy_SVM(Xvalid ,Yvalid, W, GDparam.margin)
    acc_train(1,i+1) = ComputeAccuracy_SVM(Xtrain, Ytrain, W, GDparam.margin)
    %acc_valid(1,i+1) = ComputeAccuracy_SVM2(Xvalid, ytrain, Yvalid, W, GDparam.margin)
    %acc_train(1,i+1) = ComputeAccuracy_SVM2(Xtrain, ytrain, Ytrain, W, GDparam.margin)
end

%J_test = ComputeCost_SVM(Xtest, Ytest, W, b, lambda)
acc_test = ComputeAccuracy(Xtest,ytest,W,b)

figure(1); %plot accuracy
plot([0:1:GDparam.epoch],acc_train*100,'r',[0:1:GDparam.epoch],acc_valid*100,'b');
legend({'training accuracy','validation accuracy'},'Location','southeast');
xlabel('epoch') 
ylabel('accuracy(%)')

figure(2); %plot cost
plot([0:1:GDparam.epoch],J_train,'r',[0:1:GDparam.epoch],J_valid,'b');
legend({'training loss','validation loss'},'Location','southwest');
xlabel('epoch') 
ylabel('Loss')