function [resultX,resultY] = MakeCompen(X,Y,K,min_class)
KK=size(X,1);
resultX1 = zeros(KK,min_class);resultY1 = zeros(K,min_class);
resultX2 = zeros(KK,min_class);resultY2 = zeros(K,min_class);
resultX3 = zeros(KK,min_class);resultY3 = zeros(K,min_class);
resultX4 = zeros(KK,min_class);resultY4 = zeros(K,min_class);
resultX5 = zeros(KK,min_class);resultY5 = zeros(K,min_class);
resultX6 = zeros(KK,min_class);resultY6 = zeros(K,min_class);
resultX7 = zeros(KK,min_class);resultY7 = zeros(K,min_class);
resultX8 = zeros(KK,min_class);resultY8 = zeros(K,min_class);
resultX9 = zeros(KK,min_class);resultY9 = zeros(K,min_class);
resultX10 = zeros(KK,min_class);resultY10 = zeros(K,min_class);
resultX11 = zeros(KK,min_class);resultY11 = zeros(K,min_class);
resultX12 = zeros(KK,min_class);resultY12 = zeros(K,min_class);
resultX13 = zeros(KK,min_class);resultY13 = zeros(K,min_class);
resultX14 = zeros(KK,min_class);resultY14 = zeros(K,min_class);
resultX15 = zeros(KK,min_class);resultY15 = zeros(K,min_class);
resultX16 = zeros(KK,min_class);resultY16 = zeros(K,min_class);
resultX17 = zeros(KK,min_class);resultY17 = zeros(K,min_class);
resultX18 = zeros(KK,min_class);resultY18 = zeros(K,min_class);

cnt=ones(K,1);
for i = 1: size(X,2)
    if Y(1,i) == 1
        if cnt(1,1)<min_class+1
            resultX1(:,cnt(1,1)) = X(:,i);
            resultY1(:,cnt(1,1)) = Y(:,i);
            cnt(1,1)=cnt(1,1)+1;
        end
    elseif Y(2,i) == 1
        if cnt(2,1)<min_class+1
            resultX2(:,cnt(2,1)) = X(:,i);
            resultY2(:,cnt(2,1)) = Y(:,i);
            cnt(2,1)=cnt(2,1)+1;
        end
    elseif Y(3,i) == 1
        if cnt(3,1)<min_class+1
            resultX3(:,cnt(3,1)) = X(:,i);
            resultY3(:,cnt(3,1)) = Y(:,i);
            cnt(3,1)=cnt(3,1)+1;
        end
    elseif Y(4,i) == 1
        if cnt(4,1)<min_class+1
            resultX4(:,cnt(4,1)) = X(:,i);
            resultY4(:,cnt(4,1)) = Y(:,i);
            cnt(4,1)=cnt(4,1)+1;
        end
    elseif Y(5,i) == 1
        if cnt(5,1)<min_class+1
            resultX5(:,cnt(5,1)) = X(:,i);
            resultY5(:,cnt(5,1)) = Y(:,i);
            cnt(5,1)=cnt(5,1)+1;
        end
    elseif Y(6,i) == 1
        if cnt(6,1)<min_class+1
            resultX6(:,cnt(6,1)) = X(:,i);
            resultY6(:,cnt(6,1)) = Y(:,i);
            cnt(6,1)=cnt(6,1)+1;
        end
    elseif Y(7,i) == 1
        if cnt(7,1)<min_class+1
            resultX7(:,cnt(7,1)) = X(:,i);
            resultY7(:,cnt(7,1)) = Y(:,i);
            cnt(7,1)=cnt(7,1)+1;
        end
    elseif Y(8,i) == 1
        if cnt(8,1)<min_class+1
            resultX8(:,cnt(8,1)) = X(:,i);
            resultY8(:,cnt(8,1)) = Y(:,i);
            cnt(8,1)=cnt(8,1)+1;
        end
    elseif Y(9,i) == 1
        if cnt(9,1)<min_class+1
            resultX9(:,cnt(9,1)) = X(:,i);
            resultY9(:,cnt(9,1)) = Y(:,i);
            cnt(9,1)=cnt(9,1)+1;
        end
    elseif Y(10,i) == 1
        if cnt(10,1)<min_class+1
            resultX10(:,cnt(10,1)) = X(:,i);
            resultY10(:,cnt(10,1)) = Y(:,i);
            cnt(10,1)=cnt(10,1)+1;
        end
    elseif Y(11,i) == 1
        if cnt(11,1)<min_class+1
            resultX11(:,cnt(11,1)) = X(:,i);
            resultY11(:,cnt(11,1)) = Y(:,i);
            cnt(11,1)=cnt(11,1)+1;
        end
    elseif Y(12,i) == 1
        if cnt(12,1)<min_class+1
            resultX12(:,cnt(12,1)) = X(:,i);
            resultY12(:,cnt(12,1)) = Y(:,i);
            cnt(12,1)=cnt(12,1)+1;
        end
    elseif Y(13,i) == 1
        if cnt(13,1)<min_class+1
            resultX13(:,cnt(13,1)) = X(:,i);
            resultY13(:,cnt(13,1)) = Y(:,i);
            cnt(13,1)=cnt(13,1)+1;
        end
    elseif Y(14,i) == 1
        if cnt(14,1)<min_class+1
            resultX14(:,cnt(14,1)) = X(:,i);
            resultY14(:,cnt(14,1)) = Y(:,i);
            cnt(14,1)=cnt(14,1)+1;
        end
    elseif Y(15,i) == 1
        if cnt(15,1)<min_class+1
            resultX15(:,cnt(15,1)) = X(:,i);
            resultY15(:,cnt(15,1)) = Y(:,i);
            cnt(15,1)=cnt(15,1)+1;
        end
    elseif Y(16,i) == 1
        if cnt(16,1)<min_class+1
            resultX16(:,cnt(16,1)) = X(:,i);
            resultY16(:,cnt(16,1)) = Y(:,i);
            cnt(16,1)=cnt(16,1)+1;
        end
    elseif Y(17,i) == 1
        if cnt(17,1)<min_class+1
            resultX17(:,cnt(17,1)) = X(:,i);
            resultY17(:,cnt(17,1)) = Y(:,i);
            cnt(17,1)=cnt(17,1)+1;
        end
    elseif Y(18,i) == 1
        if cnt(18,1)<min_class+1
            resultX18(:,cnt(18,1)) = X(:,i);
            resultY18(:,cnt(18,1)) = Y(:,i);
            cnt(18,1)=cnt(18,1)+1;
        end
    end
end
resultX=cat(2,resultX1,resultX2);resultY=cat(2,resultY1,resultY2);
resultX=cat(2,resultX,resultX3);resultY=cat(2,resultY,resultY3);
resultX=cat(2,resultX,resultX4);resultY=cat(2,resultY,resultY4);
resultX=cat(2,resultX,resultX5);resultY=cat(2,resultY,resultY5);
resultX=cat(2,resultX,resultX6);resultY=cat(2,resultY,resultY6);
resultX=cat(2,resultX,resultX7);resultY=cat(2,resultY,resultY7);
resultX=cat(2,resultX,resultX8);resultY=cat(2,resultY,resultY8);
resultX=cat(2,resultX,resultX9);resultY=cat(2,resultY,resultY9);
resultX=cat(2,resultX,resultX10);resultY=cat(2,resultY,resultY10);
resultX=cat(2,resultX,resultX11);resultY=cat(2,resultY,resultY11);
resultX=cat(2,resultX,resultX12);resultY=cat(2,resultY,resultY12);
resultX=cat(2,resultX,resultX13);resultY=cat(2,resultY,resultY13);
resultX=cat(2,resultX,resultX14);resultY=cat(2,resultY,resultY14);
resultX=cat(2,resultX,resultX15);resultY=cat(2,resultY,resultY15);
resultX=cat(2,resultX,resultX16);resultY=cat(2,resultY,resultY16);
resultX=cat(2,resultX,resultX17);resultY=cat(2,resultY,resultY17);
resultX=cat(2,resultX,resultX18);resultY=cat(2,resultY,resultY18);
