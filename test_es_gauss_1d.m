%
% Test of esest(.) code's ability to recover a 1D Gaussian posterior
% "exactly" in the limit of large ensemble (it should be able to do this).
%

% priorxv ~ N(mu0, sigma0^2)
sigma0 = 0.7;
mu0 = -1.3;

% measurement precision v ~ N(0, sigmav^2)
sigmav = 0.30;

% draw a sample measurement y = x + v from prior
xtrue = (sigma0*randn + mu0);
y = xtrue + sigmav*randn;

% evaluate the posterior density x|y ~ N(mu1, sigma1^2)
k = sigma0^2/(sigma0^2 + sigmav^2);
e = y - mu0;
mu1 = mu0 + k*e;
sigma1 = sqrt((1 - k))*sigma0;

% Display what happened using 1d normal density
% p(x|m,s)=exp(-0.5*(x-m)^2/s^2)/(s*sqrt(2*pi))

px = @(x, m, s)(exp(-0.5*((x - m)/s).^2)/(s*sqrt(2*pi)));

nxx = 1e3;
xx = linspace(mu0-4*sigma0, mu0+4*sigma0, nxx);
p0 = px(xx, mu0, sigma0);
pl = px(xx, y, sigmav);  % likelihood
py = px(xx, xtrue, sigmav);  % indicate the possible measurements
p1 = px(xx, mu1, sigma1);  % posterior

figure;
hold on;
plot(xx, p0, 'Color', 'k', 'LineWidth', 3);
plot(xx, py, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 3);
plot(xx, pl, 'Color', [0.7, 0.7, 1.0], 'LineWidth', 3);
plot(xx, p1, 'Color', 'b', 'LineWidth', 3);
A = axis();
line(xtrue*[1, 1], [A(3), A(4)], 'LineWidth', 3, 'LineStyle', '--', 'Color', 'c');
line(y*[1, 1], [A(3), A(4)], 'LineWidth', 3, 'LineStyle', '-.', 'Color', 'm');
axis([xx(1), xx(end), A(3), A(4)]);
legend('prior', 'possible obsv.', 'likelihood', 'posterior', 'true x', 'observation');
xlabel('x', 'FontSize', 16);
ylabel('density p(x)', 'FontSize', 16);
title('Exact Gaussian posterior', 'FontSize', 16);

esoo = esest();
esoo.EnsembleSize = 2e3;
esoo.Iterations = 8;
% All these algorithms should solve the present estimation problem:
%   'es', 'es-svd', 'es-mda', 'ies', 'es-mda-svd', 'ies-sub'
esoo.Algorithm = 'es-mda-svd';
esoo.VectorizedF = true;

% Next provide forward model f(x) = x, observation error Covv = sigmav^2,
% and prior N(mu0, Cov0=sigma0^2) to the "ensemble estimator" esest(.);

f = @(x)(x); 
Covv = sigmav*sigmav;
mu_1 = mu0;
Cov_1 = sigma0*sigma0;
xinit = [];

rep = esest(f, y, Covv, mu_1, Cov_1, xinit, esoo);

figure;
hold on;
plot(xx, p0, 'Color', 'c', 'LineWidth', 3);
plot(xx, p1, 'Color', 'b', 'LineWidth', 3);
mi = mean(rep.X);
stdi = std(rep.X);
p1_i = px(xx, mi, stdi);  % posterior ensemble final state
plot(xx, p1_i, 'Color', 'r', 'LineWidth', 3);
for ii = 1:numel(rep.Xitr)
  if isempty(rep.Xitr{ii}), continue; end
  % Show histograms of ensemble progressions here; or reductions to sample mean/variances ?  
  mi = mean(rep.Xitr{ii});
  stdi = std(rep.Xitr{ii});
  p1_i = px(xx, mi, stdi);  % posterior progressions
  plot(xx, p1_i, 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);
end
legend('true prior', 'true posterior', sprintf('final (%s)', upper(esoo.Algorithm)));
xlabel('x', 'FontSize', 16);
ylabel('density p(x)', 'FontSize', 16);
title(sprintf('Ensemble-based posterior (size = %i)', length(rep.X)), 'FontSize', 16);

