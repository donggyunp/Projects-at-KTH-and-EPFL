%data augmentation
%for 0,8 flip left right and up down
%for 3 flip ip down

%first try ,mutiplification

[Xtrain1,Ytrain1,ytrain1]=LoadBatch('Datasets/cifar-10-batches-mat/data_batch_1');
[Xtrain2,Ytrain2,ytrain2]=LoadBatch('Datasets/cifar-10-batches-mat/data_batch_2');
[Xtrain3,Ytrain3,ytrain3]=LoadBatch('Datasets/cifar-10-batches-mat/data_batch_3');
[Xtrain4,Ytrain4,ytrain4]=LoadBatch('Datasets/cifar-10-batches-mat/data_batch_4');
[Xtrain5,Ytrain5,ytrain5]=LoadBatch('Datasets/cifar-10-batches-mat/data_batch_5');

Xtrain = cat(2,Xtrain1,Xtrain2);
Xtrain = cat(2,Xtrain,Xtrain3);
Xtrain = cat(2,Xtrain,Xtrain4);
Xtrain = cat(2,Xtrain,Xtrain5(:,1:9000));
Ytrain = cat(2,Ytrain1,Ytrain2);
Ytrain = cat(2,Ytrain,Ytrain3);
Ytrain = cat(2,Ytrain,Ytrain4);
Ytrain = cat(2,Ytrain,Ytrain5(:,1:9000));
ytrain = cat(2,ytrain1,ytrain2);
ytrain = cat(2,ytrain,ytrain3);
ytrain = cat(2,ytrain,ytrain4);
ytrain = cat(2,ytrain,ytrain5(:,1:9000));

for i=1:size(Xtrain,2)
    if ytrain(1,i)=0
        