RNN = RNN_class();

m=100; K=80; epoch=7;
RNN.seq_length = 25;
RNN.eta = .3;
RNN.Initialize(m,K);

%check random generation
%{
pp=[.2, .3, .5, .3, .6;
    .45, .3, .1, .3, .2;
    .35, .4, .4, .4, .2];
syn_Y=synthesize(pp);
[val,argmax] = max(syn_Y);
for i = 1:size(argmax,2)
    charac=ind_to_char(argmax(i)+10)
end
%}
smooth_loss=0;
loss=zeros(1,1);

for j=1:epoch
    for e=1 : fix(size(book_data,2) / RNN.seq_length)
        if e == 1
            RNN.h_t(:,RNN.seq_length+1)=zeros(m,1);
            
        end
        if mod(e,10000)==1
            X_chars_s = book_data(e : e + 199);
            X_onehot_s = zeros(K,length(X_chars_s));
            for i = 1:length(X_chars_s)
                X_onehot_s(char_to_ind(X_chars_s(i)),i) = 1;
            end
            P = RNN.forward_s(X_onehot_s,hprev);
            str="";
            [val,argmax] = max(P)
            for i = 1:size(argmax,2)
                charac=ind_to_char(argmax(i));
                str = append(str,charac);
            end
            sample=str
        end
        X_chars = book_data((e-1)*RNN.seq_length + 1: ...
            e * RNN.seq_length);
        Y_chars = book_data((e-1)*RNN.seq_length + 2: ...
            e * RNN.seq_length + 1);
        X_onehot = zeros(K,length(X_chars));
        Y_onehot = zeros(K,length(X_chars));
        for i = 1:length(X_chars)
            X_onehot(char_to_ind(X_chars(i)),i) = 1;
            Y_onehot(char_to_ind(Y_chars(i)),i) = 1;
        end
        %h_t(:,1) = hprev;
        RNN.forward(X_onehot);
        %hprev=RNN.h_t(:,RNN.seq_length + 1);
        loss_train = RNN.ComputeLoss(Y_onehot);
        smooth_loss = .999 * smooth_loss + .001 * loss_train;
        loss(end+1)= smooth_loss;
        %we need gradients of V,c,W,U,b
        RNN.backprop(X_onehot,Y_onehot);
        
    end
end