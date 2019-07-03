function rep = esest(f, y, Covv, mu1, Cov1, xinit, opts)
%
% function rep = esest(f, y, Covv, mu1, Cov1, xinit, opts)
%
% Ensemble Smoother approach to (essentially) solving
% weighted least-squares type (perturbed) optimization problems
%
%   min_x { ey'*inv(Covv)*ey + ex'*inv(Cov1)*ex } 
%
% where ey = y - f(x), ex = x - mu1, for an ensemble
%
%   X = [x(1), ..., x(m)]   (m columns)
%
% The purpose is to take a prior in the form of either mu1, Cov1,
% or an explicit ensemble xinit; and then given the observation y
% and the measurement error (additive) covariance Covv, to return
% an "approximate posterior" ensemble X.
%
% f(.) is the forward problem and it can be a complete black box
% and no derivatives of it (Jacobian, Hessian) are required.
% In principle the sampling of f(.) should be parallelized.
% f(.) can be stochastic also.
%
% If xinit = [] then the prior ensemble is automatically drawn
% from N(mu1, Cov1). If xinit is non-empty it will be taken
% as the prior ensemble "as is". If mu1 = 'resample' then
% Cov1 should specify an ensemble to sample from.
%
% These algorithms are implemented (opts.Algorithm):
%   'es', 'es-mda', 'ies', 'es-svd', 'es-mda-svd', 'ies-sub', 'es-et'
%
% IES* is basically the Newton method with configurable step sizes.
% ES-MDA* is using tempered transitions (recursive appl. of ES*).
% ES* is non-iterative (one-step).
%
% The iterated algorithms IES, ES-MDA are supposed to
% better handle nonlinear problems by taking a sequence of 
% smaller updates with new "ensemble linearizations" along
% the way (compared to plain-vanilla ES).
%
% Obtain default options struct: eseo = esest();
%

%
% REFERENCES:
%
% P. Sakov and P. Oke, Monthly Weather Review, Vol 136, 2007
% G. Eversen, Computational Geosciences (2018) 22:885–908
% G. Eversen, et al., EnKF workshop presentation 2019
%

% TODO: pre-or-post update inflation options ?
% TODO: various diagnostics for ensemble "health", resampling, outlier removal ?
% TODO: improve basic IES with square roots / svd [optionally at least]
% TODO: improve IES-SUB (awkward inverse)
% TODO: es-det [Sakov/Oke] algorithm [not so critical] & random rotations

if nargin == 0
  assert(nargout <= 1);
  rep = getDefaultOptions();
  return;
end

assert(nargout <= 1, 'maximum 1 output argument');
assert(nargin == 7, 'exactly 7 (or 0) input arguments are required');

assert(isstruct(opts), 'options struct must be provided (opts).');
assert(isa(f, 'function_handle'));
assert(isvector(y), 'y should be a numeric vector');
[ny, ncolsy] = size(y);
assert(all(isfinite(y(:))), 'y must not contain non-finite numbers');
assert(ncolsy == 1, 'y must be a column vector');  % Should I allow multicolumn processing?

L1 = [];
if ischar(mu1)
  assert(strcmpi(mu1, 'resample'), 'mu1 only has 1 valid string value');
  X1 = Cov1;
  m1 = size(X1, 2);  % does not have to be the same as the ensemble size
  nx = size(X1, 1);
  mu1 = mean(X1, 2);
  A1 = X1 - repmat(mu1, [1, m1]);
  % It might be OK to ignore the regularizer ep1 by default if m1 > nx
  L1 = chol(opts.ep1*eye(nx) + (A1*A1')/(m1-1), 'lower');
elseif isvector(mu1)
  mu1 = mu1(:);
  nx = length(mu1);
  L1 = getCholeskyFactor(Cov1, nx, true);
end

if ~isempty(xinit)
  m = size(xinit, 2);
  if ~isempty(opts.EnsembleSize) && opts.EnsembleSize ~= m
    warning('opts.EnsembleSize overridden by explicit xinit column count');
  end
  X = xinit;  % assign initial ensemble X directly from xinit
  nx = size(X, 1);
  if ~isempty(mu1) || ~isempty(L1)
    warning('xinit specifies initial ensemble; mu1, Cov1 should have been set to empty');
  end
else
  assert(isvector(mu1) && ~isempty(L1));
  assert(~isempty(opts.EnsembleSize));
  m = opts.EnsembleSize;
  % Auto-generate m-member initial ensemble from N(mu1, L1*L1')
  X = drawFromMVN(m, mu1, L1);
end

Lv = getCholeskyFactor(Covv, ny, true);

assert(ischar(opts.Algorithm), 'opts.Algorithm must be a string');

if ~opts.VectorizedF
  % create a "vectorization wrapping"
  FX = @(x)(ApplyFuncToColumns(f, x));
else
  % otherwise assume f is taking matrix arguments already
  FX = f;
end

rep = struct;
rep.creator = mfilename();
rep.opts = opts;

% Allocate an error log for each output vs. iteration
rep.rmse = [];
if opts.LogRMSE
  rep.rmse = NaN(ny, opts.Iterations);
end

rep.Xitr = {};
if opts.LogIntermediate
  rep.Xitr = cell(1, opts.Iterations);
end

rep.grad = [];
if length(opts.Algorithm) >= 3 && strcmpi(opts.Algorithm(1:3), 'ies')
  rep.grad = NaN(1, opts.Iterations);   % logged for IES*
end

W = [];

if strcmpi(opts.Algorithm, 'es')

  % This is a ***one-step*** classic Kalman update of the ensemble

  if opts.LogIntermediate, rep.Xitr{1} = X; end
  Y = whitenColumns(FX(X), Lv);
  ym = mean(Y, 2);
  ay = Y - repmat(ym, [1, m]);
  xm = mean(X, 2);
  ax = X - repmat(xm, [1, m]);
  Cxy = (ax*ay')/(m - 1);
  Cyy = (ay*ay')/(m - 1);
  Kxy = Cxy/(Cyy + eye(ny));
  E = repmat(whitenColumns(y, Lv), [1, m]) - Y;  % innovation/error
  X = X + Kxy*(E + randn(ny, m));  % perturbed observation ensemble update

elseif strcmpi(opts.Algorithm, 'es-svd')

  % Same as 'es' but never explicitly forms covariance matrices

  if opts.LogIntermediate, rep.Xitr{1} = X; end
  Y = whitenColumns(FX(X), Lv);
  ym = mean(Y, 2);
  ays = (Y - repmat(ym, [1, m]))/sqrt(m - 1);
  xm = mean(X, 2);
  axs = (X - repmat(xm, [1, m]))/sqrt(m - 1);
  [Uk, Sk, Vk] = svd(ays, 'econ');
  svk = diag(Sk);
  Qk = diag(svk./(1+svk.*svk));
  E = repmat(whitenColumns(y, Lv), [1, m]) - Y;
  X = X + axs*(Vk*(Qk*(Uk'*(E + randn(ny, m)))));

elseif strcmpi(opts.Algorithm, 'es-et')

  % same as 'es' but with nonperturbed update (using sym. ensemble transform)
  % costs *** much more *** for large ensembles but presumably more precise for small ensembles
  if opts.LogIntermediate, rep.Xitr{1} = X; end
  xm = mean(X, 2);
  As = (X - repmat(xm, [1, m]))/sqrt(m - 1);
  Y = FX(X);
  ym = mean(Y, 2);
  HAs = (Y - repmat(ym, [1, m]))/sqrt(m - 1);
  % QR factorization to obtain Kalman gain
  if isvector(Lv)
    [~, tmp] = qr([HAs'; diag(Lv)], 0);
  else
    [~, tmp] = qr([HAs'; Lv'], 0);
  end
  Lk = tmp(1:ny, 1:ny)'; % Lk1*Lk1' = Lv*Lv' + HAs*HAs'
  Kk = (As*(HAs'/Lk'))/Lk;  % Kalman gain for updating ensemble mean below
  % SVD to generate ensemble transform (for anomalies As)
  Z = whitenColumns(HAs, Lv);
  [Ut, St, Vt] = svd([eye(m); Z], 'econ');
  st = diag(St);
  Tk = Vt*diag(1./st(:))*Vt';  % symmetric transform (note that no random rotation is done)
  xa = xm + Kk*(y - ym);
  Aa = sqrt(m - 1)*As*Tk;
  X = repmat(xa, [1, m]) + Aa;

elseif strcmpi(opts.Algorithm, 'ies')

  rep.gami = MakeSchemeIES(...
    opts.iesabc(1), opts.iesabc(2), opts.iesabc(3), ...
    opts.Iterations);

  % This version of IES is best used for large ensembles: m > nx

  X0 = X;  % remember inital ensemble; it is part of the cost function
  x0 = mean(X0, 2);
  A0s = (X0 - repmat(x0, [1, m]))/sqrt(m - 1);
  P0 = A0s*A0s'; % I should form regularized inverse instead really (SVD of A0s)
  Q = P0\eye(nx);  % no squaring up please; fix this.. same for linearization below..

  infQ = max(abs(Q(:)));
  % NOTE: it may be more clever to work with the square root of Q (but slower)

  yobs = whitenColumns(y, Lv);
  D = randn(ny, m);
  Ytilde = repmat(yobs, [1, m]) + D;

  for ii = 1:opts.Iterations
    if opts.LogIntermediate, rep.Xitr{ii} = X; end 
    Y = whitenColumns(FX(X), Lv);
    ym = mean(Y, 2);
    ay = Y - repmat(ym, [1, m]);
    xm = mean(X, 2);
    ax = X - repmat(xm, [1, m]);
    Cyx = (ay*ax')/(m - 1);
    Cxx = (ax*ax')/(m - 1);
    H = Cyx/Cxx;  % "ensemble linearization";
    EY = Y - Ytilde;
    E0 = X - X0; %repmat(x0, [1, m]);
    g1 = Q*E0 + H'*EY;
    g2 = Q + H'*H;
    X = X - rep.gami(ii)*(g2\g1);
    % average 2-norm of ensemble gradients of objective
    %rep.grad(ii) = mean(sqrt(sum(g1.^2, 1)/nx));
    if opts.LogRMSE
      E = repmat(yobs, [1, m]) - Y;  % pre-update ensemble unpert. output error 
      rep.rmse(:, ii) = sqrt(sum(E.^2, 2)/m);
    end
    rep.grad(ii) = max(max(abs(g1(:)))) / max([1, infQ]);
    if rep.grad(ii) < opts.MinGradientNorm
      break;
    end
  end

elseif strcmpi(opts.Algorithm, 'ies-sub')

  % Ensamble subspace version of IES (maybe more useful for large parameter spaces x).
  % Matrices are dimensioned by ensemble; so not appropriate for very large ensembles.
  % Typical use case: m < nx

  rep.gami = MakeSchemeIES(...
    opts.iesabc(1), opts.iesabc(2), opts.iesabc(3), ...
    opts.Iterations);

%  D = repmat(y, [1, m]) + drawFromMVN(m, [], Lv);  % perturbed measurement ensemble
  yobs = whitenColumns(y, Lv);
  D = repmat(yobs, [1, m]) + randn(ny, m);
  W = zeros(m, m);

  xm = mean(X, 2);
  A = X - repmat(xm, [1, m]);

  for ii = 1:opts.Iterations
    if opts.LogIntermediate, rep.Xitr{ii} = X; end
  %  Y = FX(X);
    Y = whitenColumns(FX(X), Lv);
    ym = mean(Y, 2);
    Ays = (Y - repmat(ym, [1, m]))/sqrt(m - 1);
    wm = mean(W, 2);
    Aws = (W - repmat(wm, [1, m]))/sqrt(m - 1);
    Omg = eye(m) + Aws;
    Si = Ays/Omg; %Si = ((Omg')\(Ays'))';
    Hi = Si*W + D - Y;
  %  W = W - rep.gami(ii)*(W - ((Si')/(Si*Si' + Lv*Lv'))*Hi);
    W = W - rep.gami(ii)*(W - ((Si')/(Si*Si' + eye(ny)))*Hi);
    X = repmat(xm, [1, m]) + A*(eye(m) + W/sqrt(m-1));
    g1 = W - Si'*(D - Y); % g2 = eye(nw) + (Si'*Si);
    if opts.LogRMSE
      E = repmat(yobs, [1, m]) - Y;  % pre-update ensemble unpert. output error 
      rep.rmse(:, ii) = sqrt(sum(E.^2, 2)/m);
    end
    rep.grad(ii) = max(abs(g1(:))) / (1 + max(abs(Si(:)))^2);
    if rep.grad(ii) < opts.MinGradientNorm
      break;
    end
  end

elseif strcmpi(opts.Algorithm, 'es-mda')

  % This is quite elegant in principle; it should also be implemented
  % as an optional algorithm inside the enkfest(.) code; ES-MDA stepping updates

  rep.alfi = MakeSchemeESMDA(opts.alfgeo, opts.Iterations);  % generate fixed schedule of "step-sizes"
  yobs = whitenColumns(y, Lv);

  for ii = 1:opts.Iterations
    % store each pre-update ensemble "as is"
    if opts.LogIntermediate, rep.Xitr{ii} = X; end 
    % augmented forward function / whitened
    Y = whitenColumns(FX(X), Lv);
    % determine location/spread of ensembles X and Y
    ym = mean(Y, 2);
    ay = Y - repmat(ym, [1, m]);
    xm = mean(X, 2);
    ax = X - repmat(xm, [1, m]);
    Cxy = (ax*ay')/(m - 1);
    Cyy = (ay*ay')/(m - 1);
    Kxy = Cxy/(Cyy + rep.alfi(ii)*eye(ny));
    % make stochastic update with random output perturbations
    E = repmat(yobs, [1, m]) - Y;  % innovation/error
    D = randn(ny, m);  % Yes it has to be a new realization each update
    X = X + Kxy*(E + sqrt(rep.alfi(ii))*D);
    if opts.LogRMSE
      rep.rmse(:, ii) = sqrt(sum(E.^2, 2)/m);  % store pre-update error for each iteration
    end
  end

elseif strcmpi(opts.Algorithm, 'es-mda-svd')

  rep.alfi = MakeSchemeESMDA(opts.alfgeo, opts.Iterations);  % generate fixed schedule of "step-sizes"
  yobs = whitenColumns(y, Lv);

  % Same as ES-MDA above but "non-squared" 

  for ii = 1:opts.Iterations
    if opts.LogIntermediate, rep.Xitr{ii} = X; end
    Y = whitenColumns(FX(X), Lv);
    ym = mean(Y, 2);
    ays = (Y - repmat(ym, [1, m]))/sqrt(m - 1);
    xm = mean(X, 2);
    axs = (X - repmat(xm, [1, m]))/sqrt(m - 1);
    [Uk, Sk, Vk] = svd(ays/sqrt(rep.alfi(ii)), 'econ');
    svk = diag(Sk);
    Qk = diag(svk./(1+svk.*svk))/sqrt(rep.alfi(ii));
    E = repmat(yobs, [1, m]) - Y;  % innovation/error
    Di = sqrt(rep.alfi(ii))*randn(ny, m);
    X = X + axs*(Vk*(Qk*(Uk'*(E + Di))));
    if opts.LogRMSE
      rep.rmse(:, ii) = sqrt(sum(E.^2, 2)/m);
    end
  end

elseif strcmpi(opts.Algorithm, 'es-mda-et')

  % TODO: ensemble transform version of 'es-mda'

  error('not yet implemented');

else
  error(sprintf('unrecognized algorithm: %s', opts.Algorithm));
end

% Return final ensemble and its output (if requested)
rep.X = X;
rep.Y = [];
if opts.ReturnFinalOutput
  rep.Y = FX(X);
  % evaluate final RMSE (posterior predictive)
  E = whitenColumns(repmat(y, [1, m]) - rep.Y, Lv);
  rep.rmse_pp = sqrt(sum(E.^2, 2)/m);
end

end

% SUBPROGRAMS FOLLOW BELOW

function o = getDefaultOptions()
  o = struct;
  o.EnsembleSize = 100;      % we want this to be significantly larger than x if feasible..
  o.ep1 = 0;
  o.VectorizedF = false;     % Can we provide matrix arguments to f(.) ? 
  o.Algorithm = 'es';
  o.Iterations = 10;  % common option for all iterative algorithms (fixed number of iters, or maximum)
%  o.iesabc = [0.6, 0.3, 2.5];  % IES Newton step scheme (see below)
  o.iesabc = [0.60, 0.95, 2.50];
  o.alfgeo = 1.0;  % parameter that controls the step-sequence for ES-MDA
  o.LogIntermediate = true;
  o.LogRMSE = true;
  o.ReturnFinalOutput = true;
  o.MinGradientNorm = 1e-8;  % stop before o.Iterations if IES* algorithms ?
end

% Step sequence for IES; start with step a, approach step b at a rate set by c > 1
function gami = MakeSchemeIES(a, b, c, n)
  assert(a > 0 && b > 0 && c > 1);
  gami = NaN(n, 1);
  for ii = 1:n
    gami(ii) = b + (a - b) * 2^(-(ii - 1)/(c - 1));
  end
end

% Step sequence for ES-MDA
function alfi = MakeSchemeESMDA(alfgeo, n)
  assert(alfgeo > 0 && n >= 1);
  alfi = NaN(n, 1);
  alfi(1) = 1.0;
  for ii = 2:n
    alfi(ii) = alfi(ii - 1) / alfgeo;
  end
  alfi = alfi * sum(1./alfi);
end

% "Whiten" a vector/matrix x using Cholesky factor Lx or a diagonal of standard deviations
function Z = whitenColumns(X, Lx)
  if isvector(Lx)
    Z = X./repmat(Lx(:), [1, size(X, 2)]);
  else
    Z = Lx\X;
  end
end

% Draw k samples from multivariate normal ~ N(mx, Lx*Lx')
function X = drawFromMVN(k, mx, Lx)
  if isempty(mx)
    if isvector(Lx)
      d = length(Lx);
      X = repmat(Lx(:), [1, k]).*randn(d, k);
    else
      d = size(Lx, 1);
      X = Lx*randn(d, k);
    end
  else
    assert(isvector(mx));
    d = length(mx);
    X = repmat(mx(:), [1, k]);
    if isvector(Lx)
      X = X + repmat(Lx(:), [1, k]).*randn(d, k);
    else
      X = X + Lx*randn(d, k);
    end
  end
end

% Requires M to be pos. def.
function L = getCholeskyFactor(M, n, keepVector)
  if numel(M) == 1
    if keepVector
      assert(M > 0, 'M > 0 failed');
      L = repmat(sqrt(M), [n, 1]); % Return diagonal only
      return;
    end
    M = diag(repmat(M, [n, 1]));
  elseif isvector(M) && numel(M) == n
    if keepVector
      assert(all(M > 0), 'all(M > 0) failed');
      L = sqrt(M(:)); % Return diagonal only
      return;
    end
    M = diag(M(:));
  elseif all(size(M) == [n, n])
    % good as is; nothing to do; assume M is symmetric
  else
    error('getCholeskyFactor: M, n not dimensionally compatible');
  end
  L = chol(M, 'lower');  % M = L*L'
end

% Manual apply f(.) to columns as "backup vectorization"
function fX = ApplyFuncToColumns(f, X)
  [rx, cx] = size(X);
  fX = NaN(rx, cx);
  for cc = 1:cx
    fX(:, cc) = f(X(:, cc));
  end
end
