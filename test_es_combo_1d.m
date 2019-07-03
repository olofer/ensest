%
% Test program for a nonlinear stochastic posterior sampling problem
% approximated with ES, ES-MDA (one-dimensional).
% Visualize with histograms and use HUGE ensembles m ~ 1e4+
%
% Runs with both OCTAVE and MATLAB.
%
% TASK: observation function f(x) = h(g(x) + w) where w ~ N(0, Covw)
%       and both g(.) and h(.) are nonlinear;
%
% We observe y = f(x) + v, v ~ N(0, Covv); and we have a prior ensemble X for x
% then what is the posterior ensemble X|Y=y ?
%
% Consider the random variate Y:
%   Y = h(g(X) + W) + V, where V ~ N(0, sigmav^2), X ~ Prior on X, W ~ N(0, sigmaw^2),
%   and h(.), g(.) arbitrary scalar functions
%
% The joint distribution can be written like so (latent Z, output Y, input X):
%
%   p(X=x, Y=y, Z=z) = P(X=x) * N(z - g(x)|0, sigmaw^2) * N(y - h(z)|0, sigmav^2)
%
% Recall Bayes: P(A|B) = P(A) * P(B|A) / P(B) = prior * likelihood / evidence
% And evidence P(B) = int_A dA * P(A)P(B|A); marginalize out A
%
% Here we want to estimate the density p(x|Y=y) = p(x)*p(Y=y|x)/p(Y=y)
% 

h = @(z)(exp(z));
g = @(x)(-x.^3);
sigmav = 0.1;
sigmaw = 0.05;
ab0 = [-1, +1];  % uniform prior range for Pr(X)

% Draw an observation of Y according to the probabilistic model
%xtrue = 0.8437;
xtrue = rand * diff(ab0) + ab0(1);
ztrue = g(xtrue) + sigmaw*randn;
yobs = h(ztrue) + sigmav*randn;

% Show the prior distribution of Y before the observation also
nsamples = 2e5;
yprior = h(g(rand(nsamples, 1) * diff(ab0) + ab0(1)) + sigmaw*randn(nsamples, 1)) + sigmav*randn(nsamples, 1);

figure;
hold on;
[nn, yy] = hist(yprior, 100);
pyy = nn/trapz(yy, nn);
plot(yy, pyy, 'LineWidth', 2);
A = axis();
line(yobs*[1, 1], A(3:4), 'LineStyle', '--', 'LineWidth', 2, 'Color', 'r');
legend('prior marginal Pr(Y)', 'Observation');
xlabel('y');
ylabel('Pr(Y=y)');
title('Prior observation density');

% Now we ask for the posterior density of X given Y=y.
% There are several ways to do this: integrate out the latent Z for example
%
% Let Z = g(X) + W, then Y = f(Z) + V; and express the
% joint distribution as a factorization
%   p(X,Y,Z,V,W) = p(V)*p(W)*p(X)*p(Z|X,W)*p(Y|Z,V)
%
% Conditionals can be obtained via marginalization.
%
% p(X|Y) = p(X,Y) / p(Y) = int_{Z,V,W} p(..) / int_{X,Z,V,W} p(..)
%

% grid over x for posterior evaluation
nx = 500;
xx = linspace(ab0(1) - 0.1, ab0(2) + 0.1, nx);
xx = xx(:);
% create a joint sample of V, W
vw = randn(nsamples, 2).*[repmat(sigmav, [nsamples, 1]), repmat(sigmaw, [nsamples, 1])];
pxy = zeros(nx, 1);
px0 = zeros(nx, 1);
for ii = 1:nx
  % evaluate p(X,Y,Z,V,W) = p(V)*p(W)*p(X)*p(Z|X,W)*p(Y|Z,V)
  % "integrated" over V, W, Z at X=x, Y=y using sampling
  if xx(ii) > ab0(1) && xx(ii) < ab0(2)
    px0(ii) = 1/diff(ab0);
    % sample z, and then evaluate the likelihood of Y=y, using the conditionals
    z = g(xx(ii)) + vw(:, 2);
    e = (yobs - h(z))/sigmav;  % never actually need the samples of V here
    pyzv = exp(-0.5*e.^2);
    pxy(ii) = mean(pyzv) * px0(ii);  % unnormalized is OK
  end
end

py = trapz(xx, pxy);

figure;
hold on;
plot(xx, px0, 'LineWidth', 2, 'Color', 'c');
plot(xx, pxy/py, 'LineWidth', 2, 'Color', 'b');
A = axis();
line(xtrue*[1, 1], A(3:4), 'LineStyle', '--', 'LineWidth', 2, 'Color', 'r');
title('Posterior density of X');
xlabel('x');
ylabel('Pr(X=x|Y=yobs)');
legend('prior', 'posterior', 'true x');

% Next attempt posterior approximation using ensemble smoother
m = 2e4;  % ensemble size to use
eff = @(x)(h(g(x) + sigmaw*randn(size(x))));
Covv = sigmav^2;
mu1 = [];
Cov1 = [];
xinit = rand(1, m)*diff(ab0) + ab0(1);  % initial ensemble from uniform prior
esoo = esest();
esoo.EnsembleSize = m;
esoo.VectorizedF = true;

% ES-MDA posterior approximation
esoo.Iterations = 12;
esoo.Algorithm = 'es-mda';
rep_mda = esest(eff, yobs, Covv, mu1, Cov1, xinit, esoo);
[nn_mda, xx_mda] = hist(rep_mda.X, 50);

% ES (1 step Kalman) posterior approximation
esoo.Algorithm = 'es';
rep_es = esest(eff, yobs, Covv, mu1, Cov1, xinit, esoo);
[nn_es, xx_es] = hist(rep_es.X, 50);

% IES posterior (notice that IES might have extra difficulty since f is stochastic)
esoo.Iterations = 12;  % allow more iterations
esoo.Algorithm = 'ies';
esoo.iesabc = [0.6, 0.3, 2.5];
rep_ies = esest(eff, yobs, Covv, mu1, Cov1, xinit, esoo);
[nn_ies, xx_ies] = hist(rep_ies.X, 50);

figure;
hold on;
plot(xx, pxy/py, 'LineWidth', 2, 'Color', 'b');
plot(xx_mda, nn_mda/trapz(xx_mda, nn_mda), 'LineWidth', 2, 'Color', 'm');
plot(xx_es, nn_es/trapz(xx_es, nn_es), 'LineWidth', 2, 'Color', 'r');
plot(xx_ies, nn_ies/trapz(xx_ies, nn_ies), 'LineWidth', 2, 'Color', 'c');
title(sprintf('Ensemble (size = %i) posterior approximations', m));
xlabel('x');
ylabel('Pr(X=x|Y=yobs)');
legend('posterior', 'ES-MDA posterior', 'ES posterior', 'IES posterior');

% Next visualize the posterior predictive density + ensemble approximations

% Now I can use (xx, Fxx) to sample x from the posterior
Fxx = cumtrapz(xx, pxy/py);
%i1 = find(Fxx > 0, 1);
%i2 = find(Fxx >= 1, 1);
%if isempty(i2), i2 = numel(Fxx); end
%i12 = i1:i2;

isOctave = (exist('OCTAVE_VERSION', 'builtin') ~= 0);
if isOctave
  [Fxxu, tmp] = unique(Fxx(:), 'last');
  xxu = xx(tmp);
else
  [Fxxu, tmp] = unique(Fxx(:), 'stable');
  xxu = xx(tmp);
end

% Draw samples
%xpost = interp1(Fxx(i12), xx(i12), rand(nsamples, 1));
xpost = interp1(Fxxu, xxu, rand(nsamples, 1));
ypost = h(g(xpost(:)) + sigmaw*randn(nsamples, 1)) + sigmav*randn(nsamples, 1);

[nny_es, yy_es] = hist(rep_es.Y, 50);
[nny_ies, yy_ies] = hist(rep_ies.Y, 50);
[nny_mda, yy_mda] = hist(rep_mda.Y, 50);
[nn_ypp, yy_ypp] = hist(ypost, 100);

figure;
hold on;
plot(yy, pyy, 'LineWidth', 2, 'Color', 'k');
plot(yy_ypp, nn_ypp/trapz(yy_ypp, nn_ypp), 'LineWidth', 2, 'Color', 'b');
plot(yy_mda, nny_mda/trapz(yy_mda, nny_mda), 'LineWidth', 2, 'Color', 'm');
plot(yy_es, nny_es/trapz(yy_es, nny_es), 'LineWidth', 2, 'Color', 'r');
plot(yy_ies, nny_ies/trapz(yy_ies, nny_ies), 'LineWidth', 2, 'Color', 'c');
A = axis();
line(yobs*[1, 1], A(3:4), 'LineStyle', '--', 'LineWidth', 2, 'Color', 'r');
title(sprintf('Ensemble (size = %i) post. pred. approximations', m));
xlabel('y');
ylabel('Pr[Ynext=y|Y=yobs]');
legend('prior pred.', 'post. pred.', 'ES-MDA pp', 'ES pp', 'IES pp', 'observation');
