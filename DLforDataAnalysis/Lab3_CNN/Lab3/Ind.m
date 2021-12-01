function result = Ind(X)
for i=1:size(X,1)
    for j = 1:size(X,2)
        if X(i,j) > 0
            X(i,j)=1;
        else X(i,j)=0;
        end
    end
end
result = X;