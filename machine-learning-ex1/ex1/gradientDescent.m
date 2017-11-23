function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
  
    delta = X * theta - y;
    theta0 = theta(1,1) - alpha * (1/m) * (delta' * X(:, 1));
    theta1 = theta(2,1) - alpha * (1/m) * (delta' * X(:, 2));

    theta = [theta0; theta1];

    % ============================================================

    % Save the cost J in every iteration   
    costJ = computeCost(X, y, theta);
    J_history(iter) = costJ;

end

end
