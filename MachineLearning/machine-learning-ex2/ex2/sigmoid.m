function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% if "col == 1", then "g = 1 ./ (1+exp(-z))"
col = size(z, 2);
for iter = 1:col % each columns
    g(:,iter) = 1 ./ (1+exp(-z(:,iter)));
end

% =============================================================

end
