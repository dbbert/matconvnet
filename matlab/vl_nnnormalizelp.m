function y = vl_nnnormalizelp(x,dzdy,varargin)
%VL_NNNORMALIZELP  CNN Lp normalization
%   Y = VL_NNNORMALIZELP(X) normalizes in Lp norm each spatial
%   location in the array X:
%
%       Y(i,j,k) = X(i,j,k) / sum_q (X(i,j,q).^p + epsilon)^(1/p)
%
%   DZDX = VL_NNNORMALIZELP(X, DZDY) computes the derivative of the
%   function with respect to X projected onto DZDY.
%
%   VL_NNNORMALIZE(___, 'opts', val, ...) takes the following options:
%
%   `p`:: 2
%      The exponent of the Lp norm. Warning: currently only even
%      exponents are supported.
%
%   `epsilon`: 0.01
%      The constant added to the sum of p-powers before taking the
%      1/p square root (see the formula above).
%
%   See also: VL_NNNORMALIZE().

opts.epsilon = 1e-2 ;
opts.p = 2 ;
opts = vl_argparse(opts, varargin) ;

% L1 normalization
if opts.p == 1
  mass = sum(abs(x),3) ;
  y = bsxfun(@rdivide, x, mass + opts.epsilon) ;
  
%   assert(all(sum(abs(y(1,1,:,:)),3))) ;
    
  if nargin < 2 || isempty(dzdy)
      return ;
  else  
      % TODO: check if this is correct
      dzdy = bsxfun(@rdivide, dzdy, mass) ;
      y = dzdy - bsxfun(@times, sum(dzdy .* x, 3), bsxfun(@rdivide, sign(x), mass + opts.epsilon)) ;
  end  
  
% Other normalizations
else
    massp = (sum(x.^opts.p,3) + opts.epsilon) ;
    mass = massp.^(1/opts.p) ;
    y = bsxfun(@rdivide, x, mass) ;

    if nargin < 2 || isempty(dzdy)
      return ;
    else
      dzdy = bsxfun(@rdivide, dzdy, mass) ;
      y = dzdy - bsxfun(@times, sum(dzdy .* x, 3), bsxfun(@rdivide, x.^(opts.p-1), massp)) ;
    end
end
