classdef RNN_class < handle
    properties
        W
        U
        b
        h_t
        V
        c
        P
        a_t
        o
        M_W
        M_U
        M_V
        M_c
        M_b
        eta
        seq_length
    end
    
    methods
        function Init(obj,W,U,b,h_t,V,c,P,a_t,o,M_U,M_W,M_V,M_b,M_c,eta,seq_length)
            obj.W = W;
            obj.U = U;
            obj.b =b;
            obj.h_t =h_t;
            obj.V =V;
            obj.a_t =a_t;
            obj.o =o;
            obj.c =c;
            obj.P = P;
            obj.M_W = M_W;
            obj.M_U = M_U;
            obj.M_V = M_V;
            obj.M_c = M_c;
            obj.M_b = M_b;
            obj.eta =eta;
            obj.seq_length = seq_length;
            
        end
       
        function Initialize(obj,m,K)
            sig = 0.01;
            obj.b = zeros(m,1);
            obj.c = zeros(K,1);
            obj.a_t = zeros(m,obj.seq_length);
            obj.h_t = zeros(m,obj.seq_length+1);
            obj.U = randn(m, K)*sig;
            obj.W = randn(m, m)*sig;
            obj.V = randn(K, m)*sig;
            obj.M_W = 0;
            obj.M_U = 0;
            obj.M_V = 0;
            obj.M_c = 0;
            obj.M_b = 0;

        end
        
        function forward(obj,X)
            tau = size(X,2);
            obj.h_t = zeros(size(obj.b,1),obj.seq_length+1);
            obj.h_t(:,1) = obj.h_t(:,tau+1);
            for t = 1:tau
                obj.a_t(:,t) = obj.W * obj.h_t(:,t) + obj.U * X(:,t) + obj.b;
                obj.h_t(:,t+1) = tanh(obj.a_t(:,t));
                obj.o = obj.V * obj.h_t(:,t+1) + obj.c;
                obj.P(:,t) = softmax(obj.o);
            end
        end
       
        function P = forward_s(obj,X,hprev)
            tau = size(X,2);
            a_t = zeros(size(obj.a_t,1),tau);
            h_t = zeros(size(obj.h_t,1),tau+1);
            h_t(:,1) = hprev;
            for t = 1:tau
                a_t(:,t) = obj.W * h_t(:,t) + obj.U * X(:,t) + obj.b;
                h_t(:,t+1) = tanh(a_t(:,t));
                o = obj.V * h_t(:,t+1) + obj.c;
                P(:,t) = softmax(o);
            end
        end
        
        function loss = ComputeLoss(obj,Y)
            tau = size(Y,2);
            loss = 0;
            for t=1:tau
                loss = loss -sum(log(Y(:,t)'*obj.P(:,t)));
            end
        end
        
        function backprop(obj,X,Y)
            K = size(X,1);
            tau = size(X,2);
            %dLdl_t = 1;
            dLdp_t = zeros(K,tau);
            dLdo_t = zeros(K,tau);
            dLdV = zeros(size(obj.V));
            %need dLdc
            for t = 1:tau %maybe t = tau:1 ?????
                dLdp_t(:,t) = -Y(:,t)/(Y(:,t)'*obj.P(:,t));
                dLdo_t(:,t) = -(Y(:,t)-obj.P(:,t));
                %dLdp_t = max(min(dLdp_t, 5), -5);
                %dLdo_t = max(min(dLdo_t, 5), -5);
                dLdV = dLdV + dLdo_t(:,t) * obj.h_t(:,t+1)';%?? dLdo_t=g_t
            end
            dLdc = sum(dLdo_t,2);
            
            %dLdV = max(min(dLdV, 5), -5);
            obj.M_V = obj.M_V + sum(sum(dLdV.^2));
            obj.V = obj.V - obj.eta/((obj.M_V+1e-6)^(1/2)) * dLdV;
            
            %dLdc = max(min(dLdc, 5), -5);
            obj.M_c = obj.M_c + sum(sum(dLdc.^2));
            obj.c = obj.c - obj.eta/((obj.M_c+1e-6)^(1/2)) * dLdc;
            
            dLdh_t = zeros(size(obj.h_t));
            dLdh_t(:,tau+1) = obj.V' * dLdo_t(:,tau);%??
            %dLdh_t = max(min(dLdh_t, 5), -5);
            dLda_t=zeros(size(obj.a_t));
            dLda_t(:,tau) = diag(1-(obj.h_t(:,tau+1).^2)) * dLdh_t(:,tau+1);
            %dLda_t = max(min(dLda_t, 5), -5);
            %from here, h=1~tau(h_t:2~tau+1) ,a=1:tau-1 in backward recursively
            for t = tau-1:-1:1
                dLdh_t(:,t+1) = obj.V' * dLdo_t(:,t) + obj.W*dLda_t(:,t+1);
                %dLdh_t = max(min(dLdh_t, 5), -5);
                dLda_t(:,t) = diag(1-(obj.h_t(:,t+1).^2)) * dLdh_t(:,t+1);
                %dLda_t = max(min(dLda_t, 5), -5);
            end
            dLdW = 0;
            dLdU = 0;
            dLdb = 0;
            for t=1:tau
                dLdW = dLdW + dLda_t(:,t)*obj.h_t(:,t)';
                dLdU = dLdU + dLda_t(:,t)*X(:,t)';
                dLdb = dLdb + diag(1-obj.h_t(:,t+1).^2)*dLdh_t(:,t+1);
            end
            dLdW = max(min(dLdW, 5), -5);
 
            %dLdU = max(min(dLdU, 5), -5);
            %dLdb = max(min(dLdb, 5), -5);
            obj.M_W = obj.M_W + sum(sum(dLdW.^2));
            obj.M_U = obj.M_U + sum(sum(dLdU.^2));
            obj.M_b = obj.M_b + sum(sum(dLdb.^2));
            obj.W = obj.W - obj.eta/((obj.M_W+1e-6)^(1/2)) * dLdW;
            obj.U = obj.U - obj.eta/((obj.M_U+1e-6)^(1/2)) * dLdU;
            obj.b = obj.b - obj.eta/((obj.M_b+1e-6)^(1/2)) * dLdb;
        end
        
        function [loss,acc] = Compute_train(obj,X,Y)
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
        
    end
end