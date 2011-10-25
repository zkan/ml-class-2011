function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
grad = zeros(size(theta));

for iter = 1:num_iters

    h = sigmoid(X * theta);
    
    for i=1:size(theta, 1)
      theta(i) = theta(i) - (alpha * (1 / m) * (h - y)' * X(:, i));
    end

    % ============================================================

    % Save the cost J and grad in every iteration    
    [J_history(iter), grad] = costFunction(theta, X, y);

end

end
