function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% sum along dimension 2 (add each column together)
z = sum(X*theta, 2);

hx = sigmoid(z);
diff = -y.*log(hx) - (1-y).*log(1-hx);
J = sum(diff)/m + sum(theta.*theta)*lambda/m;

grad = sum((hx - y).*X)/m;
grad = grad';
grad(2:end,:) = grad(2:end,:) + theta(2:end,:)*lambda/(m-1);

% =============================================================

end
