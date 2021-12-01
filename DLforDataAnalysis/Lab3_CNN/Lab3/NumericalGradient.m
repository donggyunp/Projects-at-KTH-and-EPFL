function Gs = NumericalGradient(X_inputs, Ys, ConvNet, h)

try_ConvNet = ConvNet;
Gs = cell(length(try_ConvNet.F)+1, 1);

for l=1:length(try_ConvNet.F)
    try_ConvNet.F{l} = try_ConvNet.F{l};
    
    Gs{l} = zeros(size(try_ConvNet.F{l}));
    nf = size(try_ConvNet.F{l},  3);
    
    for i = 1:nf        
        try_ConvNet.F{l} = try_ConvNet.F{l};
        F_try = squeeze(try_ConvNet.F{l}(:, :, i));
        G = zeros(numel(F_try), 1);
        
        for j=1:numel(F_try)
            F_try1 = F_try;
            F_try1(j) = F_try(j) - h;
            try_ConvNet.F{l}(:, :, i) = F_try1; 
            
            [try_ConvNet,l1] = try_ConvNet.ComputeLoss(X_inputs, Ys);
            
            F_try2 = F_try;
            F_try2(j) = F_try(j) + h;            
            
            try_ConvNet.F{l}(:, :, i) = F_try2;
            [try_ConvNet,l2] = try_ConvNet.ComputeLoss(X_inputs, Ys);            
            
            G(j) = (l2 - l1) / (2*h);
            try_ConvNet.F{l}(:, :, i) = F_try;
        end
        Gs{l}(:, :, i) = reshape(G, size(F_try));
    end
end

%% compute the gradient for the fully connected layer
W_try = ConvNet.W;
G = zeros(numel(W_try), 1);
for j=1:numel(W_try)
    W_try1 = W_try;
    W_try1(j) = W_try(j) - h;
    try_ConvNet.W = W_try1; 
            
    [try_ConvNet,l1] = try_ConvNet.ComputeLoss(X_inputs, Ys);
            
    W_try2 = W_try;
    W_try2(j) = W_try(j) + h;            
            
    try_ConvNet.W = W_try2;
    [try_ConvNet,l2] = try_ConvNet.ComputeLoss(X_inputs, Ys);            
            
    G(j) = (l2 - l1) / (2*h);
    try_ConvNet.W = W_try;
end
Gs{end} = reshape(G, size(W_try));