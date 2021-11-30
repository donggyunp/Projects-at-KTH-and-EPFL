function [grad_W] = ComputeGradient_SVM(X, Y, W, lambda, margin)

eval = W*X;
grad_W = zeros(size(W,1),size(X,1));
count = 0;
%grad_b = zeros(size(W,1),1);
Y=int32(Y);
[argvalue, argmax] = max(Y);
for j = 1:size(X,2)
    for i = 1:size(W,1)   
        if i == argmax(1,j) %correct class row
            for k = 1:size(W,1)
                if eval(i,j)-eval(argmax(1,j),j)+margin > 0 && k ~= argmax(1,j)
                    count = count + 1;
                end
            end
            grad_W(argmax(1,j),:) = -count*X(:,j)';
        else                %wrong class row
            if eval(i,j)-eval(argmax(1,j),j)+margin > 0
                grad_W(i,:) = X(:,j)';
            end
        end
    end
end


%grad_W = grad_L_W + 2*lambda.*W;
