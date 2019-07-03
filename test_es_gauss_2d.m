%
% Test of esest(.) code's ability to recover a 2D Gaussian posterior
% "exactly" in the limit of large ensemble (it should be able to do this).
%
% The exact result (for comparison) is obtained
% from a linear-Gaussian system relation [e.g. Section 4.4 Murphy 2012].
%

xtrue = (2*rand(2, 1) - 1);  % uniform true location x in [-1, +1]

% Prior may or may not be off-target
mu0 = randn(2, 1);
sigma0 = 1.0;

% Generate random projection measurement matrix A (each row has unit length)
k = 9;
A = randn(k, 2);
A = A./repmat(sqrt(sum(A.^2, 2)), [1, 2]);
sigmav = 0.30;  % measurement noise

% Make noisy observations
y = A*xtrue + sigmav*randn(k, 1);

% Calculate the exact posterior distribution for x|y, denote mu1, Cov1
Cov1 = (eye(2)/sigma0^2 + (A'*A)/(sigmav^2))\eye(2);
mu1 = Cov1*(A'*y/sigmav^2 + mu0/sigma0^2);

% Estimate posterior using esest(.)
% rep = esest(f, y, Covv, mu1, Cov1, xinit, opts)
f = @(x)(A*x);  % the linear observation function
m = 2e3;  % ensemble size

esoo = esest();
esoo.EnsembleSize = m;
esoo.VectorizedF = true;
%esoo.iesabc = [0.95 0.95 2.00]; % this will converge pretty fast
esoo.iesabc = [0.60, 0.95, 2.50];

esoo.Algorithm = 'es';
estA = esest(f, y, sigmav^2, mu0, sigma0.^2, [], esoo);  % basic ES 1-step solution

esoo.Algorithm = 'es-mda-svd';  % or 'es-mda'
estB = esest(f, y, sigmav^2, mu0, sigma0.^2, [], esoo);

esoo.Algorithm = 'ies'; %'ies-sub';
estC = esest(f, y, sigmav^2, mu0, sigma0.^2, [], esoo);

esoo.Algorithm = 'es-svd';
estD = esest(f, y, sigmav^2, mu0, sigma0.^2, [], esoo);

esoo.Algorithm = 'es-et';
estE = esest(f, y, sigmav^2, mu0, sigma0.^2, [], esoo);

% Make a plot of (mu0, Cov0) and (mu1, Cov1) and the true x

xlvl = 2;  % ~90% level ellipses
npts = 200;

Xprior = repmat(mu0, [1, m]) + sigma0*randn(2, m);
ellipse0 = makeEllipseFromEnsemble(Xprior, npts, xlvl);

Lpost = chol(Cov1, 'lower');
Xpost = repmat(mu1, [1, m]) + Lpost*randn(2, m);
ellipse1 = makeEllipseFromEnsemble(Xpost, npts, xlvl);

ellipseA = makeEllipseFromEnsemble(estA.X, npts, xlvl);
ellipseB = makeEllipseFromEnsemble(estB.X, npts, xlvl);
ellipseC = makeEllipseFromEnsemble(estC.X, npts, xlvl);
ellipseD = makeEllipseFromEnsemble(estD.X, npts, xlvl);
ellipseE = makeEllipseFromEnsemble(estE.X, npts, xlvl);

figure;
hold on;
axis equal;
plot(ellipse0(:, 1), ellipse0(:, 2), 'Color', 'c', 'LineWidth', 3);
plot(ellipse1(:, 1), ellipse1(:, 2), 'Color', 'b', 'LineWidth', 3);
plot(xtrue(1), xtrue(2), ...
  'Marker', 'o', 'MarkerSize', 10, 'Color', 'k', 'LineStyle', 'none');
plot(ellipseA(:, 1), ellipseA(:, 2), 'Color', 'm', 'LineWidth', 2);
plot(ellipseB(:, 1), ellipseB(:, 2), 'Color', 'r', 'LineWidth', 2);
plot(ellipseC(:, 1), ellipseC(:, 2), 'Color', 'g', 'LineWidth', 2);
plot(ellipseD(:, 1), ellipseD(:, 2), 'Color', 'y', 'LineWidth', 2);
plot(ellipseE(:, 1), ellipseD(:, 2), 'Color', 'm', 'LineWidth', 2, 'LineStyle', '--');
for ii = 1:numel(estB.Xitr)
  if isempty(estB.Xitr{ii}), continue; end
  ellipseB = makeEllipseFromEnsemble(estB.Xitr{ii}, npts, xlvl);
  plot(ellipseB(:, 1), ellipseB(:, 2), 'Color', [1.0, 0.7, 0.7], 'LineWidth', 1);
end
for ii = 1:numel(estC.Xitr)
  if isempty(estC.Xitr{ii}), continue; end
  ellipseC = makeEllipseFromEnsemble(estC.Xitr{ii}, npts, xlvl);
  plot(ellipseC(:, 1), ellipseC(:, 2), 'Color', [0.7, 1.0, 0.7], 'LineWidth', 1);
end
xlabel('X1', 'FontSize', 16);
ylabel('X2', 'FontSize', 16);
hl = legend(...
  'prior', ...
  'posterior', ...
  'true x', ...
  upper(estA.opts.Algorithm), ...
  upper(estB.opts.Algorithm), ...
  upper(estC.opts.Algorithm), ...
  upper(estD.opts.Algorithm), ...
  upper(estE.opts.Algorithm), ...
  sprintf('steps for %s', upper(estB.opts.Algorithm)));
set(hl, 'FontSize', 12);
grid on;
title(sprintf('#ensemble = %i, #observation = %i', m, k), 'FontSize', 16);
