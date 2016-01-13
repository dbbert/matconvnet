function Y = vl_nnrepellingloss(X,dzdy,varargin)
%VL_NNLOSS CNN categorical or attribute loss.
%   Y = VL_NNLOSS(X, C) computes the loss incurred by the prediction
%   scores X given the categorical labels C.

% X = rand(1,1,3,10);
X = gather(X); % TODO: remove this when backwards pass is vectorized.
opts.instanceWeights = [] ;
opts = vl_argparse(opts,varargin) ;

inputSize = [size(X,1) size(X,2) size(X,3) size(X,4)] ;

% assert(inputSize(1) == 1) ;
% assert(inputSize(2) == 1) ;

% --------------------------------------------------------------------
% Spatial weighting
% --------------------------------------------------------------------
batchSize = inputSize(4);
margin = 2*batchSize^2;

% L1 normalization
% X = bsxfun(@rdivide, X, sum(X,3) + eps) ;

% distanceMatrix = dist(squeeze(X(1,1,:,:))); % take the pairwise euclidean (L2) distances
distanceMatrix = mandist(squeeze(X(1,1,:,:))); % take the pairwise manhattan (L1) distances
    
if nargin <= 1 || isempty(dzdy) % forward    
    distanceSum = sum(distanceMatrix(:));
    loss = distanceSum;
%     loss = max(0, margin - distanceSum) ;
    Y = loss ;
%     Y = loss / batchSize ;    
else % backward
%     t0 = tic;
    Y = zerosLike(X);
%     for y = 1:inputSize(1)
%         for x = 1:inputSize(2)
%             for i = 1:batchSize
%                 a = X(y,x,:,i);
%                 sumd = 0;
%                 for j = 1:batchSize
%                     b = X(y,x,:,j);
%                     d = a - b;
% %                     sumd = sumd + d; % L2
%                   sumd = sumd + sign(d); % L1
%                 end
%                 Y(y,x,:,i) = sumd ;
%             end
%         end
%     end
%     toc(t0)
    
%     t1 = tic;
    % Thanks Otto Debals!
%     M = -ones(batchSize)+batchSize*eye(batchSize);
%     Y = tmprod(X,M,4);
%     toc(t1)
% %     t2 = tic;
% %     M = -ones(batchSize)+batchSize*eye(batchSize);
% %     Y2 = permute(squeeze(X)*M,[3 4 1 2]);
% %     toc(t2)
%     
%     frob(Y - Y1)
% %     frob(Y - Y2)
% %     frob(Y1 - Y2)

for i = 1:batchSize
    A = X(:,:,:,i);
    tmp1 = sum(bsxfun(@gt,A,X),4); % checking the number of elements greater than the current element
    tmp2 = sum(bsxfun(@lt,A,X),4); % checking the number of elements smaller than the current element
    Y(:,:,:,i) = tmp1-tmp2;
end
    
%     frob(Y - Y1)

    Y = bsxfun(@times, dzdy, -Y) ;
    
end

Y = gpuArray(Y); % TODO: remove this when backwards pass is vectorized.

% --------------------------------------------------------------------
function clippedGradient = clipGradient(gradient, maxNorm)
% --------------------------------------------------------------------
clippedGradient = gradient;
gradientNorm = norm(squeeze(gradient), 2);
if gradientNorm > maxNorm
    clippedGradient = gradient * maxNorm/gradientNorm;
end

% --------------------------------------------------------------------
function y = zerosLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
  y = gpuArray.zeros(size(x),'single') ;
else
  y = zeros(size(x),'single') ;
end