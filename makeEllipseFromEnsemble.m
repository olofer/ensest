function [xy, th] = makeEllipseFromEnsemble(X, nth, xlvl)
%
% function [xy, th] = makeEllipseFromEnsemble(X, nth, xlvl)
%
% Obtain the contour of a confidence ellipse in 2D
% from the n-member ensemble X (2 rows, n columns).
% The contour has nth points and represents the 
% confidence levels 70, 90, or 95 percent
% (as selected by xlvl = 1, 2, or 3).
%

assert(size(X, 1) == 2, 'must be 2D ensemble');
n = size(X, 2);  % members (if too small we might get strange results?)

assert(numel(xlvl) == 1 && (xlvl == 1 || xlvl == 2 || xlvl == 3));
assert(numel(nth) == 1 && nth >= 10);

th = linspace(0, 2*pi, nth);
th = th(:);

mX = mean(X, 2);
Xts = (X - repmat(mX, [1, n]))/sqrt(n - 1);
[U, S, V] = svd(Xts*Xts');
sv = diag(S);

sclvec = sqrt([2.41, 4.605, 5.991]);
scl = sclvec(xlvl);  % 70, 90, 95 % bounds for bivariate normal distribution

xy = NaN(nth, 2);
for jj = 1:nth
  xy(jj, :) = (mX + scl*cos(th(jj))*sqrt(sv(1))*V(:, 1) + scl*sin(th(jj))*sqrt(sv(2))*V(:, 2))';
end

end
