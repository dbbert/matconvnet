function Y = vl_nnMMDloss(X,Z,dzdy,varargin)
%VL_NNLOSS CNN categorical or attribute loss.
%   Y = VL_NNLOSS(X, C) computes the loss incurred by the prediction
%   scores X given the categorical labels C.

% X = randn(1,1,3,64) ;
inputSize = [size(X,1) size(X,2) size(X,3) size(X,4)] ;
vectorSize = inputSize(3) ;
batchSize = inputSize(4) ;
% Z = gpuArray.rand(size(X,1), size(X,2), size(X,3), size(X,4), 'single') ; % TODO: put on gpuArray or so.
% Z = vl_nnnormalizelp(Z, [], 2, 'epsilon', eps) ;
% Z = ones(size(X), 'single')

% X = gpuArray.rand(size(X), 'single') ; % TODO: put on gpuArray or so.
% X = vl_nnnormalizelp(X, [], 2, 'epsilon', eps) ;

M = size(X, 4);
N = size(Z, 4);

% X = gather(X); % TODO: remove this when backwards pass is vectorized.
opts.instanceWeights = [] ;
opts = vl_argparse(opts,varargin) ;

% assert(inputSize(1) == 1) ;
% assert(inputSize(2) == 1) ;

% --------------------------------------------------------------------
% Spatial weighting
% --------------------------------------------------------------------

% We set sigma to be the median distance between points in the aggregate sample, 
% as a compromise between these two extremes: this remains a heuristic
% http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
% distMatrix =  dist(squeeze(cat(4,gather(X(1,1,:,:)),gather(Z(1,1,:,:))))) ;
distMatrix =  dist(squeeze(gather(Z(1,1,:,:)))) ;
sigma2 = median(distMatrix(:)) ;
% sigma2 = 8.25 ;

% % calculate mmd2
% sum0 = 0 ;
% for n = 1:N
%     x = squeeze(Z(:,:,:,n));
%     for n2 = 1:N
% %         if n == n2, continue; end;
%         x_ = squeeze(Z(:,:,:,n2));
%         sum0 = sum0 + k(x,x_) ;
%     end
% end
% 
% sum1 = 0 ;
% for m = 1:M
%     x = squeeze(X(:,:,:,m));
%     for m2 = 1:M
% %         if m == m2, continue; end;
%         x_ = squeeze(X(:,:,:,m2));
%         sum1 = sum1 + k(x,x_) ;
%     end
% end
% 
% sum2 = 0 ;
% for m = 1:M
%     x = squeeze(X(:,:,:,m));
%     for n = 1:N
%         x_ = squeeze(Z(:,:,:,n));
%         sum2 = sum2 + k(x,x_) ;
%     end
% end

% mmd2 = 1/(N*(N-1)) * sum0 + 1/(M*(M-1)) * sum1 - 2/(M*N) * sum2;
% mmd2 = 1/N^2 * sum0 + 1/M^2 * sum1 - 2/(M*N) * sum2;

mmd2 = 1/N^2 * sumFunction(Z,Z) + 1/M^2 * sumFunction(X,X) - 2/(M*N) * sumFunction(X,Z);
    
if nargin <= 2 || isempty(dzdy) % forward
%     Y = mean(mmd2(:)) ;
    Y = mean(sqrt(mmd2(:))) ;
else % backward
%     Y = zerosLike(X) ;
%     for i = 1:M
%         x_i_s = squeeze(X(:,:,:,i)) ;
%         sum0 = 0 ;
%         for j = 1:M
%             x_j_s = squeeze(X(:,:,:,j)) ;
%             sum0 = sum0 + k(x_i_s,x_j_s)*(x_j_s-x_i_s) ;
%         end
%         
%         sum1 = 0;
%         for j = 1:N
%             x_j_d = squeeze(Z(:,:,:,j)) ;
%             sum1 = sum1 + k(x_i_s,x_j_d)*(x_j_d-x_i_s) ;
%         end
% 
%         der = 2/sigma2 * (1/M^2 * sum0 - 1/(M*N) * sum1) ;
%         Y(1,1,:,i) = der;
%     end
    
    Y = 2/sigma2 * (1/M^2 * sumFunctionBack(X,X) - 1/(M*N) * sumFunctionBack(X,Z)) ;
    
    Y = Y .* dzdy ;
    Y = bsxfun(@times, 1./(2*sqrt(mmd2)), Y) ;
end

if any(imag(Y(:)))
    keyboard;
end
% Y = gpuArray(Y); % TODO: remove this when backwards pass is vectorized.

% --------------------------------------------------------------------
function y = k(x,x_)
% --------------------------------------------------------------------
    d = x - x_;
    y = exp(-(d'*d)/(2*sigma2));
end

function y = sumFunction(X,Y)
    sx = size(X);
    D = bsxfun(@minus,X,reshape(Y,[sx(1:end-1) 1 sx(end)]));
    D = permute(D, [1 2 4 5 3]);
    D = exp(-(sum(D.*D, 5))/(2*sigma2));
%     D = exp(-(sum(D.*D, 5))/(2*0.5)) + exp(-(sum(D.*D, 5))/(2*1)) + exp(-(sum(D.*D, 5))/(2*5)) + exp(-(sum(D.*D, 5))/(2*10));
    y = sum(sum(D, 3), 4);
end

function y = sumFunctionBack(X,Y)
    sx = size(X);
    D = bsxfun(@minus,X,reshape(Y,[sx(1:end-1) 1 sx(end)]));
    D = permute(D, [1 2 4 5 3]);
    D = exp(-(sum(D.*D, 5))/(2*sigma2));
%     D = exp(-(sum(D.*D, 5))/(2*0.5)) + exp(-(sum(D.*D, 5))/(2*1)) + exp(-(sum(D.*D, 5))/(2*5)) + exp(-(sum(D.*D, 5))/(2*10));
    
    E = bsxfun(@minus,X,reshape(Y,[sx(1:end-1) 1 sx(end)]));
    E = permute(E, [1 2 4 5 3]);
    
    y = bsxfun(@times,D,E);
    y = permute(y, [1 2 5 3 4]);
    
    y = -sum(y, 5);
end

% --------------------------------------------------------------------
function y = zerosLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
  y = gpuArray.zeros(size(x),'single') ;
else
  y = zeros(size(x),'single') ;
end
end

end