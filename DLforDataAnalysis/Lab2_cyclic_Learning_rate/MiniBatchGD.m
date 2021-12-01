function [W1star, b1star, W2star, b2star, cnt_out,l_out, cost_t,acc_t,cost_v,acc_v,track]...
= MiniBatchGD(cost_t_in,cost_v_in, acc_t_in ,acc_v_in, X, y, Y, X_v, y_v, Y_v, GDparams,cnt_in,l_in, track,W1, b1, W2, b2, lambda)

n = size(X,2);
cost_v = track;
acc_v = track;
for j = 1 : n / GDparams.size_batch
            cnt_in = 1+cnt_in;    
            
            if mod(cnt_in,2*GDparams.eta_s) == 0
                l_in = l_in + 1;
            end
            l_out = l_in;
            if (cnt_in >= 2*l_in*GDparams.eta_s) && (cnt_in < (2*l_in + 1)*GDparams.eta_s)
                GDparams.eta_t = GDparams.eta_min + (cnt_in-2*l_in*GDparams.eta_s)/GDparams.eta_s*(GDparams.eta_max-GDparams.eta_min);    
            else 
                GDparams.eta_t = GDparams.eta_max - (cnt_in-(2*l_in+1)*GDparams.eta_s)/GDparams.eta_s*(GDparams.eta_max-GDparams.eta_min);    
            end

            j_start = (j-1)*GDparams.size_batch + 1;
            j_end = j * GDparams.size_batch;
            %inds = j_start:j_end;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
            ybatch = y(:, j_start:j_end);
            Xbatch_v = X_v(:, j_start:j_end/49);
            Ybatch_v = Y_v(:, j_start:j_end/49);
            ybatch_v = y_v(:, j_start:j_end/49);
            S1 = W1 * Xbatch + b1;
            H = max(0,S1);
            P = EvaluateClassifier(Xbatch, W1, b1, W2, b2);
            [grad_W2, grad_b2] = ComputeGradient_2nd(H, Ybatch, P, W2, lambda);
            [grad_W1, grad_b1] = ComputeGradient_1st(Xbatch, Ybatch, H, P, W1, W2, lambda);
            W1 = W1 - GDparams.eta_t * grad_W1;
            b1 = b1- GDparams.eta_t * grad_b1;
            W2 = W2 - GDparams.eta_t * grad_W2;
            b2 = b2- GDparams.eta_t * grad_b2; 
            %GDparams.count = GDparams.count + 1;            
            track(1,cnt_in) = GDparams.eta_t;

            cost_t_in(1,cnt_in) = ComputeCost(Xbatch,Ybatch,W1,b1,W2,b2,lambda);
            acc_t_in(1,cnt_in) = ComputeAccuracy(Xbatch,ybatch,W1,b1,W2,b2);
            %validation with batch
            cost_v_in(1,cnt_in) = ComputeCost(Xbatch_v,Ybatch_v,W1,b1,W2,b2,lambda);
            acc_v_in(1,cnt_in) = ComputeAccuracy(Xbatch_v,ybatch_v,W1,b1,W2,b2);
            
            %validation with whole
            %cost_v_in(1,cnt_in) = ComputeCost(X_v,Y_v,W1,b1,W2,b2,lambda);
            %acc_v_in(1,cnt_in) = ComputeAccuracy(X_v,y_v,W1,b1,W2,b2);
            
            cnt_out = cnt_in;
            
            cost_t = cost_t_in;
            acc_t = acc_t_in;
            cost_v = cost_v_in;
            acc_v = acc_v_in;
end

W1star = W1;
b1star = b1;
W2star = W2;
b2star = b2;