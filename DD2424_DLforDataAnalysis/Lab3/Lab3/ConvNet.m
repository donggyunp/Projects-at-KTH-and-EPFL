classdef ConvNet < handle
    
    properties
        X1
        X2
        F1
        F2
        W
        P
        eta
        rho
        moment_W
        moment_F1
        moment_F2
    end
    
    methods
        function Init(obj,X1,X2, F1, F2, W, P, moment_W, moment_F1, moment_F2)
            obj.X1 = X1;
            obj.X2 = X2;
            obj.F1 = F1;
            obj.F2 = F2;
            obj.W = W;
            obj.P = P;
            obj.moment_W=moment_W;
            obj.moment_F1=moment_F1;
            obj.moment_F2=moment_F2;
        end
        
        function Initialize(obj,h1,k1,n1,k2,n2,K,n_len)
            sig1 = sqrt(2/h1/k1/n1); %sparse matrix => need to fix !!
            sig2 = sqrt(2/n1/k2/n2); %He Init
            obj.F1 = randn(h1, k1, n1)*sig1;
            obj.F2 = randn(n1, k2, n2)*sig2;
            n_len1 = n_len-k1+1;
            n_len2 = n_len1 -k2 +1;
            fsize=n2*n_len2;
            sig3 = sqrt(2/K/fsize);
            obj.W = randn(K, fsize)*sig3;
        end
        
        function MF1 = MakeMFMatrix1(obj, n_len)
            [dd, k, nf] = size(obj.F1);
            
            MF1 = zeros( (n_len-k+1)*nf , n_len*dd );
            F_flat = obj.F1(:)';
            
            for i = 1:(n_len-k+1)
                for j = 1:nf
                    MF1((i-1)*nf +j, (i-1)*dd+1 : (i-1)*dd + size(F_flat,2)/nf)...
                        = F_flat(1,(j-1)*dd*k+1 : j*dd*k);
                end
            end
        end
        
        function MF2 = MakeMFMatrix2(obj, n_len)
            [dd, k, nf] = size(obj.F2);
            
            MF2 = zeros( (n_len-k+1)*nf , n_len*dd );
            F_flat = obj.F2(:)';
            
            for i = 1:(n_len-k+1)
                for j = 1:nf
                    MF2((i-1)*nf +j, (i-1)*dd+1 : (i-1)*dd + size(F_flat,2)/nf)...
                        = F_flat(1,(j-1)*dd*k+1 : j*dd*k);
                end
            end
        end
        
        function forward(obj,X,n_len)
            MF1 = obj.MakeMFMatrix1(n_len);
            n_len1 = n_len-size(obj.F1,2)+1;
            MF2 = obj.MakeMFMatrix2(n_len1);
            obj.X1 = max(MF1 * X, 0);
            obj.X2 = max(MF2 * obj.X1 ,0);
            S = obj.W * obj.X2;
            obj.P = softmax(S);
        end
        
        
        function [loss,acc] = ComputeLossandAcc(obj,X,Y)
            %{
            MF1 = obj.MakeMFMatrix(obj.F1,n_len);
            n_len1 = n_len-size(obj.W,2)+1;
            MF2 = obj.MakeMFMatrix(obj.F2,n_len1);
            obj.X1 = max(MF1 * X, 0);
            %Each x(1) has size nlen1 ×1 where nlen1 = nlen −k1 +1
            %which X(1) has size n1×nlen1.
            obj.X2 = max(MF2 * obj.X1 ,0);
            S = obj.W * obj.X2;
            obj.P_batch = softmax(obj.S_batch);
            %obj.G_batch = obj.W * obj.X_batch_2;
            %}
            %loss= 1/size(X,2) * sum(sum(-log(Y' * obj.P)));
            loss= 1/size(X,2) * sum(sum(-Y' * log(obj.P)));
            [argvalue, argmax] = max(obj.P);
            [argvalue2, argmax2] = max(Y);
            acc = 0;
            for i = 1:size(X,2)
                if argmax2(1,i) == argmax(1,i)
                    acc = acc + 1;
                end
            end
            acc = acc/size(X,2)*100;
        end
        
        function [loss,acc] = Compute_valid(obj,X,Y,n_len)
            MF1 = obj.MakeMFMatrix1(n_len);
            n_len1 = n_len-size(obj.F1,2)+1;
            MF2 = obj.MakeMFMatrix2(n_len1);
            X1 = max(MF1 * X, 0);
            X2 = max(MF2 * X1 ,0);
            S = obj.W * X2;
            P = softmax(S);
            loss= 1/size(X,2) * sum(sum(-Y' * log(P)));
            %loss= 1/size(X,2) * sum(sum(-Y' * log(obj.P)));
            [argvalue, argmax] = max(P);
            [argvalue2, argmax2] = max(Y);
            acc = 0;
            for i = 1:size(X,2)
                if argmax2(1,i) == argmax(1,i)
                    acc = acc + 1;
                end
            end
            acc = acc/size(X,2)*100;
        end
        
        function backprop(obj,X,Y,n_len)
            batch =size(X,2);
            [d, k1, n1]=size(obj.F1);
            [n1, k2, n2]=size(obj.F2);
            n_len1 = n_len- k1 +1;
            n_len2 = n_len1 - k2 +1;
            dldvecF_1 = 0;
            dldvecF_2 = 0;
            G_batch = -(Y - obj.P);
            dLdW = 1/batch * G_batch * obj.X2';
            %%%%%%%%%%%%%
            obj.moment_W = obj.rho*obj.moment_W + obj.eta*dLdW;
            obj.W = obj.W -obj.moment_W;
            %%%%%%%%%%%%%
            %obj.W = obj.W - obj.eta*dLdW;
            %%%%%%%%%%%%%
            G_batch = obj.W'*G_batch;
            G_batch = G_batch.*Ind(obj.X2);
            for j = 1:batch
                x_j = obj.X1(:,j);
                g_j = G_batch(:,j);
                %v = g_j' * obj.MakeMXMatrix(x_j,size(obj.F2,1), size(obj.F2,2), size(obj.F2,3));
                
                %%%%%%%%%%%%%%%sparse version
                M = obj.MakeMFMatrix_improved(x_j);
                g_matrix = zeros(n_len2,n2);
                for t=1:n_len2
                    g_matrix(t,:) = g_j((t-1)*n2+1 : t*n2,1)';
                end
                v = M' * g_matrix;
                v = v(:);
                %%%%%%%%%%%%%%%%
                
                dldvecF_2 = dldvecF_2 + 1/batch*v;
                dldF_2 = reshape(dldvecF_2,[size(obj.F2,1), size(obj.F2,2), size(obj.F2,3)]);%how reshape? sus!!!!!
                %%%%%%%%%%%
                obj.moment_F2 = obj.rho*obj.moment_F2 + obj.eta*dldF_2;
                obj.F2 = obj.F2 - obj.moment_F2;
                %%%%%%%%%%%
                %obj.F2 = obj.F2 - obj.eta*dldF_2;
            end
            n_len1 = n_len-size(obj.F1,2)+1;
            MF2 = obj.MakeMFMatrix2(n_len1);
            G_batch = MF2' *  G_batch;
            G_batch = G_batch .* Ind(obj.X1);
            for j = 1:batch
                g_j = G_batch(:,j);
                x_j = X(:,j);
                %v = g_j' * obj.MakeMXMatrix(x_j,size(obj.F1,1), size(obj.F1,2), size(obj.F1,3));
                %%%%%%%%%%%%%%%%%
                sparsed = sparse(obj.MakeMXMatrix(x_j,d,k1,n1));
                v = g_j' * sparsed; %sparse version
                %%%%%%%%%%%%%%%%%
                dldvecF_1 = dldvecF_1 + 1/batch*v;
                dldF_1 = reshape(dldvecF_1,[size(obj.F1,1), size(obj.F1,2), size(obj.F1,3)]);
                %%%%%%%%%%%%
                obj.moment_F1 = obj.rho * obj.moment_F1 + obj.eta * dldF_1;
                obj.F1 = obj.F1 - obj.moment_F1;
                %%%%%%%%%%%%
                %obj.F1 = obj.F1 - obj.eta*dldF_1;
                %%%%%%%%%%%%
            end
            
        end
        
        function MX = MakeMXMatrix(obj,x_input, d, k, nf)
            
            n_len = size(x_input,1)/d;
            X_flat = x_input(:)';
            
            MX = zeros( (n_len-k+1)*nf, k*nf*d);
            
            for i = 1:(n_len-k+1)
                for j = 1:nf
                    MX((i-1)*nf+j, (j-1)*d*k +1 : (j-1)*d*k +d*k  ) = X_flat(1,(i-1)*d +1:(i-1)*d+d*k);
                end
            end
        end
        
        function result = MakeMFMatrix_improved(obj,X)
            [d, k1, n1]=size(obj.F1);
            [n1, k2, n2]=size(obj.F2);
            n_len = 19;
            n_len1 = n_len- k1 +1;
            n_len2 = n_len1 - k2 +1;
            X_flat = X(:)';
            F_flat = obj.F2(:)';
            result = zeros(n_len2, n1*k2);
            
            for i = 1:(n_len2)
                result(i,:) = F_flat(1,(i-1)*n1 +1:(i-1)*n1+n1*k2);
            end
        end
    end
    
end