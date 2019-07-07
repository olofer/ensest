%
% Demonstrate basic functionality of kfltvest(.) code.
%
% Define a dynamical system which is just a random walk.
% Obtain regular noisy observations (every q-th step).
%
% Show the KF (forward) and the KS (fw+backward) estimates.
%

dt = 0.02;
nt = 500;
x0 = rand;
w = sqrt(dt) * randn(nt, 1);  % Covw = dt
x = x0 + cumsum(w);
tvec = 0:(nt-1);
tvec = tvec(:)*dt;

% Generate observations y (sparse)
q = 25;
Covv = 0.2^2;
y = NaN(nt, 1);
idxy = q:q:nt;
y(idxy) = x(idxy) + sqrt(Covv)*randn(numel(idxy), 1);

% Run KF/KS with "wide-ish prior"
Cov1 = 5;
mu1 = 0;
Covw = dt;
u = zeros(nt, 1);

kfo = kfltvest();
kfo.computeSmoothedOutputs = true;

kfs = kfltvest(1, 0, 1, 0, 1, Covw, Covv, Cov1, mu1, u, y, kfo);

sigmaxf = 2*abs(kfs.Pf);
sigmaxs = 2*abs(kfs.Ps);  % since state is scalar this is enough
sigmays = 2*abs(kfs.Pys); % Plot the 2*sigma ranges for "uncertainties"

% Make plot with filter results
figure;
plot(tvec, x, 'Color', 'k', 'LineWidth', 2);
hold on;
plot(tvec, y, 'r*');
plot(tvec, kfs.xf, 'Color', 'm', 'LineWidth', 2);
plot(tvec, [kfs.xf - sigmaxf, kfs.xf + sigmaxf], 'Color', [1.0 0.8 0.8], 'LineWidth', 1);
xlabel('time t');
ylabel('random walk x(t)');
legend('latent state', 'observations', 'Kalman Filter');
title('Filtering noisy and sparse observations of a random walk');

% Make plot with smoother results
figure;
plot(tvec, x, 'Color', 'k', 'LineWidth', 2);
hold on;
plot(tvec, y, 'r*');
plot(tvec, kfs.xs, 'Color', 'b', 'LineWidth', 2);
plot(tvec, [kfs.xs - sigmaxs, kfs.xs + sigmaxs], 'Color', [0.8 0.8 1], 'LineWidth', 1);
plot(tvec, [kfs.ys - sigmays, kfs.ys + sigmays], 'Color', [0.8 0.8 0.8], 'LineWidth', 1);
xlabel('time t');
ylabel('random walk x(t)');
legend('latent state', 'observations', 'Kalman Smoother');
title('Smoothing noisy and sparse observations of a random walk');
