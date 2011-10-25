%% Initialization
clear all; close all;

data = csvread('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Choose some alpha value
alpha = 0.1;
num_iters = 5;
theta = zeros(3, 1);

[theta, J_history] = gradientDescent(X, y, initial_theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Print theta to screen
fprintf('theta: \n');
fprintf(' %f \n', theta);
