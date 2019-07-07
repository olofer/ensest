function rep = kfltvest(A, B, C, D, E, Covw, Covv, Cov1, mu1, u, y, opts)
%
% function rep = kfltvest(A, B, C, D, E, Covw, Covv, Cov1, mu1, u, y, opts)
%
% Kalman filtering, fixed-lag smoothing, and fixed-interval smoothing.
% Handles general linear time-varying dynamical systems:
%
%   x(k+1) = A(k)*x(k) + B(k)*u(k) + E(k)*w(k)
%     y(k) = C(k)*x(k) + D(k)*u(k) + v(k)
%
% where elements y(k) may contain NaNs anywhere flagging "no measurement"
% i.e. y(k), k = 1..n can have any number of elements in y(k) being NaNs.
% u(k) is required to exist at all times k = 1..n
%
% The noise covariances are stationary (expectation = E[.]):
%
%   E[x(1)*x(1)'] = Cov1, E[x(1)] = mu1   (pre-measurement @ time=1)
%   E[w(k)*w(k)'] = Covw, E[w(k)] = 0     (no correlation in time, uncorrelated with v)
%   E[v(k)*v(k)'] = Covv, E[v(k)] = 0     ( --- )
%
% If u is empty, B and D must also be empty.
% If E is empty it is set to the identity matrix so that dim(w) = dim(x)
%
% USAGE:
%
%   kfo = kfltvest();           % get default options struct
%   rep = kfltvest(..., kfo);   % run filter, smoothers dep. on kfo
%
% Cov1, Covw, Covv can be specified as scalars, vectors or matrices
% (requiring appropriate dimensions)
%
% A,B,C,D,E should be specified as callbacks (function handles) that
% returns appropriately sized matrices. Time-invariant matrices can be
% specified directly. Matrix dimensions must be time-invariant.
%
% The exact contents of the results report rep depends on the
% options struct opts. The filtered state rep.xf and its covariance
% (possibly square root of) rep.Pf will always be returned.
% The marginal log-likelihood is stored in .lliklf.
% The forecast errors (innovations) are stored in .ef.
% The Kalman Smoother results (backward pass): .xs, .Ps.
% Fixed-lag results (if enabled): .xl, .Pl (cell arrays).
% See code/comments for further details.
%

%
% *** possible improvements ***
% - compute the "smoothed" RMSE always if ys is evaluated ..
% - companion plotting tool, std-extraction tool ?
% - for fixed-lag smoothing, there is redundancy in the output
% - eliminate the storage of Pfm if the square root RTS is used (?)
% - efficient handling of diagonal covariance specification
%   (this can be very important for larger problems)
% - notice that it is possible to compute the covariances based on the non-NaN
%   patterns only; should explicitly allow this mode of operation
%   (it is already handled by simply ignoring the xf,xs outputs)
%

assert(nargout <= 1, 'too many output arguments requested');

if nargin == 0
  rep = getDefaultOptions();
  return;
end

assert(nargin == 12, 'All inputs (arguments) must be provided');

if isnumeric(A) && numel(size(A)) == 2, A = @(k)(A); end
if isnumeric(C) && numel(size(C)) == 2, C = @(k)(C); end
assert(isa(A, 'function_handle'));
assert(isa(C, 'function_handle'));

nx = size(A(1), 1);
assert(size(A(1), 2) == nx, 'A must be square matrix');
ny = size(C(1), 1);
assert(size(C(1), 2) == nx, 'ncol(C) = nx must hold');

% E can be empty so handle that with some finesse
if isempty(E), E = eye(nx); end
if isnumeric(E) && numel(size(E)) == 2, E = @(k)(E); end
assert(isa(E, 'function_handle'));
assert(size(E(1), 1) == nx, 'nrow(E) = nx must hold');
nw = size(E(1), 2);  % E is nx-by-nw, implying that Covw must be nw-by-nw (required below)

assert(isstruct(opts), 'opts argument must be a struct');

algoCholesky = (ischar(opts.algorithm) && strcmpi(opts.algorithm, 'cholesky') == 1);
algoStandard = isempty(opts.algorithm) || (ischar(opts.algorithm) && strcmpi(opts.algorithm, 'standard') == 1);
assert(algoStandard || algoCholesky, 'opts.algorithm not recognized');

assert(isnumeric(y) && numel(size(y)) == 2);
n = size(y, 1);

assert(size(y, 2) == ny, 'ncol(y) = nrow(C) must hold');

% Check that y has no Infs
assert(all(~isinf(y(:))), 'y cannot have Infs');

% If there are no NaNs we may want to optimize the algorithm loop (not done for now)
areThereNaNs = any(isnan(y(:)));
areThereOnlyNaNs = all(isnan(y(:)));

if areThereOnlyNaNs && computeBackwardPass
  error('It makes no sense to provide all-missing y AND request a backward pass');
end

if isempty(mu1), mu1 = zeros(nx, 1); end
assert(isvector(mu1) && length(mu1) == nx, 'mu1 must be a vector with length nx');

% We might want to issue a warning if all measurements are NaNs?
% But this is also a valid use-case which also should be optimized..
% predict-only mode ... i.e. extrapolate future Normal(x(f), P(f))
% distribution given the inital {x(1), Cov1(1)}

% Pre-factorize covariances: crashes unless these are pos. def.
% This is intentional: require well specified covariances.
Lw = getCholeskyFactor(Covw, nw);
Lv = getCholeskyFactor(Covv, ny);
L1 = getCholeskyFactor(Cov1, nx);

% Input specifications?
hasInput = ~isempty(u);
if hasInput
  assert(isnumeric(u) && numel(size(u)) == 2);
  assert(all(isfinite(u(:))), 'u must be finite');
  assert(size(u, 1) == n, 'nrow(u) = nrow(y) is required');
  nu = size(u, 2);
  if isempty(B), B = zeros(nx, nu); end
  if isnumeric(B) && numel(size(B)) == 2, B = @(k)(B); end
  assert(isa(B, 'function_handle'));
  assert(size(B(1), 1) == nx && size(B(1), 2) == nu, 'dim(B) error');
  if isempty(D), D = zeros(ny, nu); end
  if isnumeric(D) && numel(size(D)) == 2, D = @(k)(D); end
  assert(isa(D, 'function_handle'));
  assert(size(D(1), 1) == ny && size(D(1), 2) == nu, 'dim(D) error');
else
  nu = 0;
  assert(isempty(B) && isempty(D), 'B and D must be empty when u is empty');
end

rep = struct;
rep.creator = mfilename();
rep.opts = opts;
rep.dims_xyuw = [nx, ny, nu, nw];

% Pre-allocate required numeric arrays
% (it would be possible to not have xfm, Pfm but it speeds up the backward pass a little)
rep.xf = NaN(n, nx);       % best forward estimate using all data so far
rep.Pf = NaN(n, nx, nx);   % best forward covariance estimate 
rep.xfm = NaN(n, nx);      % pre-update estimate (different unless y(.,:) are all NaNs)
rep.Pfm = NaN(n, nx, nx);  % pre-update covariance

rep.lliklf = []; % marginal log-likelihood across forward pass
if opts.computeLogLikelihood
  rep.lliklf = NaN(n, 1);
end

rep.ef = [];
if opts.storeForwardError
  rep.ef = NaN(n, ny);
end

if opts.computeRMSE
  sumsqe = zeros(ny, 1);  % sum of squares of errors for each channel
end

if opts.fixedLagSmoothing > 0
  if opts.fixedLagSmoothing >= n
    warning('fixed-lag horizon is larger than signal y');
  end
  rep.xl = cell(n, 1);
  rep.Pl = cell(n, 1);
end

% Notice that the Covariance sequences Pf, Ps only depends on the 
% "patterns" of observation (i.e. which components of y and at what times)
% but not on the values of the measurements.

if algoStandard
  
  R = Lv*Lv';  % re-square
  Q = Lw*Lw';

  x = mu1(:);    % initial state estimate    (pre-update)
  P = L1*L1';    % initial state covariance  (pre-update)

  for kk = 1:n
    rep.xfm(kk, :) = x';
    rep.Pfm(kk, :, :) = P;
    if hasInput, uk = u(kk, :)'; end
    yk = y(kk, :)';  % yk = column vector of measurements (may contain (all) NaNs).
    iy = find(~isnan(yk));
    if ~isempty(iy)
      % Take the measurement update; yk may be partial only
      Ck = C(kk);
      Ck = Ck(iy, :); % only extract the relevant rows for this observation vector
      ek = yk(iy) - Ck*x;
      if hasInput
        Dk = D(kk);
        ek = ek - Dk(iy, :)*uk;  % adjust innovation ek if there is a direct term
      end
      % Compute Kalman gain & make updates -> {x,P} will be post-measurement variables
      Rk = R(iy, iy); % iRk = Rk\eye(length(iy));
      if opts.computeLogLikelihood
        tmp = chol(Ck*P*Ck' + Rk, 'lower');  % factorize Sk = Ck*P*Ck' + Rk = tmp*tmp'
        Kk = ((P*Ck')/tmp')/tmp;
        logpk = logliklNormal(tmp, [], ek);
        rep.lliklf(kk) = logpk;
      else
        Kk = (P*Ck')/(Ck*P*Ck' + Rk);
      end
      x = x + Kk*ek;
      %P = (P\eye(nx) + Ck'*iRk*Ck)\eye(nx);
      % Another alternative is: P = (eye(nx) - Kk*Ck)*P.
      % The "stabilized Joseph" form below maintains symmetry and pos.def.-ness
      ImKC = eye(nx) - Kk*Ck;
      P = ImKC*P*ImKC' + Kk*Rk*Kk';
      if opts.storeForwardError
        rep.ef(kk, iy) = ek';
      end
      if opts.computeRMSE
        sumsqe(iy) = sumsqe(iy) + ek.^2;
      end
    end
    % Store {x, P} in arrays xf, Pf (same as xfm, Pfm if there was no measurement at all)
    rep.xf(kk, :) = x';
    rep.Pf(kk, :, :) = P;
    if opts.fixedLagSmoothing > 0
      nlag = opts.fixedLagSmoothing;
      [xlk, Plk] = BackwardSmoothingPass( ...
        rep.xf, rep.Pf, rep.xfm, rep.Pfm, ...
        A, E, Lw, kk, nlag, opts);
      rep.xl{kk} = xlk;
      rep.Pl{kk} = Plk;
    end
    % Compute the extrapolation updates using {A, B, E, Q}
    Ak = A(kk);
    x = Ak*x;
    if hasInput
      x = x + B(kk)*uk;
    end
    Ek = E(kk);
    P = Ak*P*Ak' + Ek*Q*Ek';
  end

else

  % R is accessed only when receiving incomplete y-vectors
  % to compute a new smaller Cholesky factor used in the measurement update
  % (can this be optimized ?)
  R = Lv*Lv';

  x = mu1(:);
  L = L1;  % L = Cholesky factor of P 

  for kk = 1:n
    rep.xfm(kk, :) = x';
    rep.Pfm(kk, :, :) = L;
    if hasInput, uk = u(kk, :)'; end
    yk = y(kk, :)';
    iy = find(~isnan(yk));
    if ~isempty(iy)
      % We must do the measurement update(s)
      Ck = C(kk);
      Ck = Ck(iy, :);
      ek = yk(iy) - Ck*x;
      if hasInput
        Dk = D(kk);
        ek = ek - Dk(iy, :)*uk;
      end
      nky = length(iy);
      if nky == ny
        Lvk = Lv;
      else
        Lvk = chol(R(iy, iy), 'lower');  % is this really necessary?
      end
      if opts.squareRootJosephUpdate
        % Compute gain Kk using intermediate factorization of: Ck*P*Ck' + Rk = tmp*tmp'
        [~, Rx] = qr([L'*Ck'; Lvk'], 0);
        tmp = Rx';  % tmp is lower triangular and used to compute Kk
        %Kk = ((L*(L'*Ck'))/(tmp'))/tmp;
        Kk = (L*((L'*Ck')/(tmp')))/tmp;  % interleave multiply-then-divide
        x = x + Kk*ek; 
        % Produce update of L based on the QR representation of the Joseph form 
        [~, Rx] = qr([L'*(eye(nx) - Kk*Ck)'; Lvk'*Kk'], 0);
        L = Rx';
        if opts.computeLogLikelihood
          logpk = logliklNormal(tmp, [], ek);
          rep.lliklf(kk) = logpk;
        end
      else
        % QR factorization merging the calculation of the updated L and the nudge gain Kk
        [~, tmp] = qr([Lvk', zeros(nky, nx); L'*Ck', L']);
        assert(size(tmp, 1) == nky + nx && size(tmp, 2) == nky + nx);
        R11 = tmp(1:nky, 1:nky);
        R12 = tmp(1:nky, (nky+1):(nky+nx));
        Kk = (R12')/(R11');
        x = x + Kk*ek; 
        L = tmp((nky+1):(nky+nx), (nky+1):(nky+nx))';
        if opts.computeLogLikelihood
          logpk = logliklNormal(R11', [], ek);
          rep.lliklf(kk) = logpk;
        end
      end
      if opts.storeForwardError
        rep.ef(kk, iy) = ek';
      end
      if opts.computeRMSE
        sumsqe(iy) = sumsqe(iy) + ek.^2;
      end
    end
    % Store post-measurment {x,L} (no change if iy was empty)
    rep.xf(kk, :) = x';
    rep.Pf(kk, :, :) = L;
    if opts.fixedLagSmoothing > 0
      nlag = opts.fixedLagSmoothing;
      [xlk, Plk] = BackwardSmoothingPass( ...
        rep.xf, rep.Pf, rep.xfm, rep.Pfm, ...
        A, E, Lw, kk, nlag, opts);
      rep.xl{kk} = xlk;
      rep.Pl{kk} = Plk;
    end
    % Propagate/extrapolate expected value
    Ak = A(kk);
    x = Ak*x;
    if hasInput
      x = x + B(kk)*uk;
    end
    % Propagate/extrapolate square-root of P using "economy-size" QR factorization
    Ek = E(kk);
    [~, Rx] = qr([L'*Ak'; Lw'*Ek'], 0);
    L = Rx';
  end

end

if opts.computeRMSE
  rep.rmsy = NaN(ny, 1);  % this is useful for reference
  rep.rmse = NaN(ny, 1);
  for jj = 1:ny
    idxyj = find(isfinite(y(:, jj)));
    njj = length(idxyj);
    if njj > 0
      rep.rmsy(jj) = sqrt(mean(y(idxyj, jj).^2));
      rep.rmse(jj) = sqrt(sumsqe(jj)/njj);
    else
      assert(sumsqe(jj) == 0);
    end
  end
end

if opts.checkResults
  assert(all(isfinite(rep.xf(:))));
  assert(all(isfinite(rep.Pf(:))));
  assert(all(isfinite(rep.xfm(:))));
  assert(all(isfinite(rep.Pfm(:))));
end

if ~opts.fixedIntervalSmoothing
  return;
end

% full interval RTS Smoother recursion 
[xs, Ps] = BackwardSmoothingPass( ...
  rep.xf, rep.Pf, rep.xfm, rep.Pfm, ...
  A, E, Lw, n, n - 1, opts);

assert(size(xs, 1) == n);
assert(size(Ps, 1) == n);

rep.xs = xs;
rep.Ps = Ps;

if opts.checkResults
  assert(all(isfinite(rep.xs(:))));
  assert(all(isfinite(rep.Ps(:))));
end

if ~opts.computeSmoothedOutputs
  return;
end

rep.ys = NaN(n, ny);
rep.Pys = NaN(n, ny, ny);  % The expected (noisy) output covariance

for kk = 1:n
  xk = rep.xs(kk, :)';
  Ck = C(kk);
  if algoStandard
    thisP = squeeze(rep.Ps(kk, :, :));
    thisL = chol(thisP, 'lower');
  else
    thisL = squeeze(rep.Ps(kk, :, :));
  end
  % I can always evaluate the expected ys and its covariance Pys
  % regardless of whether a measurement exists..
  [~, tmp] = qr([thisL'*Ck'; Lv'], 0);  % Note: 0*Lv' here results in the noiseless output covariance factor
  assert(size(tmp, 1) == ny && size(tmp, 2) == ny);
  Ly = tmp'; % Now Ly*Ly' = R + Ck*Ps*Ck' =def= Pys
  yhatk = Ck*xk;
  if hasInput
    yhatk = yhatk + D(kk)*u(kk, :)';
  end
  rep.ys(kk, :) = yhatk';
  if algoStandard
    rep.Pys(kk, :, :) = Ly*Ly';
  else
    rep.Pys(kk, :, :) = Ly;
  end
end

end  % end main function

% Rauch-Tung-Striebel backward pass; start at index k and go nlag > 1 steps backwards
% return the results as xs, Ps; if k is the end point n and nlag = n then the full 
% fixed-interval Kalman Smoother result is obtained.
function [xs, Ps] = BackwardSmoothingPass(xf, Pf, xfm, Pfm, A, E, Lw, kmax, nlag, opts)
  algoStandard = ...
    isempty(opts.algorithm) || ...
    (ischar(opts.algorithm) && strcmpi(opts.algorithm, 'standard') == 1);
  n = size(xf, 1);
  nx = size(xf, 2);
  assert(nlag >= 1);
  assert(kmax >= 1 && kmax <= n);
  kmin = kmax - nlag;
  if kmin < 1, kmin = 1; end
  xs = NaN(nlag + 1, nx);
  Ps = NaN(nlag + 1, nx, nx);
  xs(nlag + 1, :) = xf(kmax, :);
  Ps(nlag + 1, :, :) = Pf(kmax, :, :);
  ll = nlag;
  if algoStandard
    for kk = (kmax - 1):-1:kmin
      Phatpred = squeeze(Pfm(kk + 1, :, :));
      Phat = squeeze(Pf(kk, :, :));
      Ak = A(kk);
      Jk = (Phat*Ak')/Phatpred;
      xhatpred = xfm(kk + 1, :)';
      xhat = xf(kk, :)';
      xs(ll, :) = (xhat + Jk*(xs(ll + 1, :)' - xhatpred))';
      Ps(ll, :, :) = Phat + Jk*(squeeze(Ps(ll + 1, :, :)) - Phatpred)*Jk';
      ll = ll - 1;
    end
  elseif opts.squareRootRTS
    % True square root backward recursion using QR factorizations
    assert(~isempty(E));
    assert(~isempty(Lw));
    for kk = (kmax - 1):-1:kmin
      Lhat = squeeze(Pf(kk, :, :));
      Ak = A(kk);
      Ek = E(kk);
      nw = size(Ek, 2);
      if nw < nx % Notice the zero-padding trick here!
        [~, tmp] = qr([[Lw'*Ek'; zeros(nx - nw, nx)], zeros(nx, nx); Lhat'*Ak', Lhat']);
      else
        assert(nw == nx, 'nw <= nx required');
        [~, tmp] = qr([Lw'*Ek', zeros(nx, nx); Lhat'*Ak', Lhat']);  % 1st QR
      end
      assert(size(tmp, 1) == 2*nx && size(tmp, 2) == 2*nx);
      R11 = tmp(1:nx, 1:nx);  % auto-detect triangularity we hope
      R12 = tmp(1:nx, (nx + 1):(2*nx));
      R22 = tmp((nx + 1):(2*nx), (nx + 1):(2*nx));
      Jk = (R12')/(R11');  % backward gain matrix
      xhatpred = xfm(kk + 1, :)';
      xhat = xf(kk, :)';
      xs(ll, :) = (xhat + Jk*(xs(ll + 1, :)' - xhatpred))';
      Ls1 = squeeze(Ps(ll + 1, :, :));  % this is the factor to be propagated backwards
      [~, tmp] = qr([Ls1'*Jk'; R22], 0);  % 2nd QR
      assert(size(tmp, 1) == nx && size(tmp, 2) == nx);
      Ps(ll, :, :) = (tmp');  % this is the new Cholesky factor !
      ll = ll - 1;
    end
  else
    % This is not a TRUE square root form; it uses the Cholesky matrices from the 
    % forward pass but squares them (symmetrically) and refactorizes at each backward step.
    for kk = (kmax - 1):-1:kmin
      Lhatpred = squeeze(Pfm(kk + 1, :, :));  % Phatpred = Lhatpred*Lhatpred'
      Lhat = squeeze(Pf(kk, :, :));           % Phat = Lhat*Lhat'
      Ak = A(kk);
      Jk = (Lhat*((Lhat'*Ak')/(Lhatpred')))/Lhatpred;  % alternating multiply-backsubstitute
      xhatpred = xfm(kk + 1, :)';
      xhat = xf(kk, :)';
      xs(ll, :) = (xhat + Jk*(xs(ll + 1, :)' - xhatpred))';
      Ls1 = squeeze(Ps(ll + 1, :, :));  % Cholesky factor to "downdate"
      tmp1 = Jk*Ls1; % split in squares to maintain symmetric ops.
      tmp2 = Jk*Lhatpred;
      thisL = chol(Lhat*Lhat' + tmp1*tmp1' - tmp2*tmp2', 'lower'); % yup, squaring up; not very pretty
      Ps(ll, :, :) = thisL;
      ll = ll - 1;
    end
  end
  assert(ll == 0);
end

% It is requires that M is positive definite, or we crash since the
% chol(.) call will issue error below (it can be made to return a flag instead if needed).
function L = getCholeskyFactor(M, n)
  if numel(M) == 1
    M = diag(repmat(M, [n, 1]));
  elseif isvector(M) && numel(M) == n
    M = diag(M(:));
  elseif all(size(M) == [n, n])
    % good as is; nothing to do
  else
    error('getCholeskyFactor: M, n not dimensionally compatible');
  end
  L = chol(M, 'lower');  % M = L*L'
end

% Given a proper Cholesky factor L of a covariance matrix P = L*L' > 0, and a mean vector mu
% consider the multivariate normal distribution N(mu, P) and evaluate the logarithm of the
% likelihood for an observation x (x and mu are k-dimensional real-valued).
% Setting mu = [], will imply mu = zeros(length(x), 1);
function logp = logliklNormal(L, mu, x)
  assert(isvector(x));
  k = length(x);
  assert(size(L, 1) == k && size(L, 2) == k);
  if isvector(mu)
    assert(length(mu) == k);
    r = L \ (x(:) - mu(:));
  else
    assert(isempty(mu));
    r = L \ x(:);
  end
  ld = abs(diag(L));  % If L comes from a QR factorization it may have minus signs on diagonal
  logp = -0.5*(r'*r) - 0.5*k*log(2*pi) - sum(log(ld));
end

% Complete set of options with default values
function o = getDefaultOptions()
  o = struct;
  o.algorithm = 'cholesky';            % can also be 'standard' (aka. '') which is faster
  o.storeForwardError = true;
  o.computeLogLikelihood = true;
  o.computeRMSE = true;
  o.fixedIntervalSmoothing = true;     % Backward pass is enabled by default
  o.computeSmoothedOutputs = false;
  o.fixedLagSmoothing = -1;            % number of time-steps backward ( <= 0 disables )
  o.squareRootJosephUpdate = false;
  o.squareRootRTS = true;              % use true square root Rauch-Tung-Striebel backward pass
  o.checkResults = true;
end
