function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
zhidden = [ones(m,1),X] * Theta1';
ahidden = sigmoid(zhidden);
zoutput = [ones(m,1),ahidden] * Theta2';
aoutput = sigmoid(zoutput);%h
%display(size(aoutput));
J = 0;
v = eye(num_labels);
y = v(y,:);%m x num_labels
J = -(sum(sum(log(aoutput).*y + log(1-aoutput).*(1-y))))/m;
regJ = lambda * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))/(2*m);
J = regJ + J;
%use back propagation to compute grad
delta3 = aoutput - y;
%step by step to compute delta
delta2 = delta3 * Theta2;
delta2 = delta2(:,2:end);
delta2 = delta2.*ahidden.*(1-ahidden);
grad1 = 1/m * delta2' * [ones(m,1),X]; %derivative about Theta1 without regularizing
grad2 = 1/m * delta3' * [ones(m,1),ahidden]; %derivative about Theta2 without regularizing
grad1reg = 1/m*[zeros(size(Theta1, 1), 1), Theta1(:,2:end)];
grad2reg = 1/m*[zeros(size(Theta2, 1), 1), Theta2(:,2:end)];
Theta1_grad = grad1 + lambda * grad1reg;
Theta2_grad = grad2 + lambda * grad2reg;
%Theta_ij means weight on j->i, if j = 0, then without regularization










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
