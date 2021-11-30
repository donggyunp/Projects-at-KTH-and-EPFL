function [X,Y,y] = LoadBatch(filename)

A = load(filename);
X = reshape(A.data',3072,10000); 
X = double(X)/255;
y = reshape(A.labels',1,10000);
y = double(y);
Y = zeros(10,10000);
for i = 1:10000
    if y(1,i)==0
        Y(1,i)=1;
    end
    if y(1,i)==1
        Y(2,i)=1;
    end
    if y(1,i)==2
        Y(2,i)=1;
    end
    if y(1,i)==3
        Y(4,i)=1;
    end
    if y(1,i)==4
        Y(5,i)=1;
    end
    if y(1,i)==5
        Y(6,i)=1;
    end
    if y(1,i)==6
        Y(7,i)=1;
    end
    if y(1,i)==7
        Y(8,i)=1;
    end
    if y(1,i)==8
        Y(9,i)=1;
    end
    if y(1,i)==9
        Y(10,i)=1;
    end
end
y = y+1;