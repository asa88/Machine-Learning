function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)



% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
%W1 is 25 64 W2 is 64 25
m=size(data,2);
weightsbuffer = ones(1,m);
a1=data; %65 10000
z2=W1*data +repmat(b1,1,m); % 25 10000
a2=sigmoid(z2); % 25 10000
z3=W2*a2 +repmat(b2,1,m);
a3=sigmoid(z3);
hx=a3; %64 10000
y=data;

rho=a2*weightsbuffer'/m;
sparsity=0;
for i=1:size(rho,1)
  sparsity=sparsity + sparsityParam* log(sparsityParam/rho(i)) + (1-sparsityParam)*log((1-sparsityParam)/(1-rho(i)));
end

regularization=lambda*(sum(sum(W1.^2)) +sum(sum(W2.^2)))/2;
cost= 	sum(sum(((hx-y).^2)/2))/m + regularization +beta*sparsity;

% b1 is 25 1 and b2 is 64 1


gradW1 = zeros(size(W1));
gradW2 = zeros(size(W2)); 
gradb1 = zeros(size(b1));
gradb2 = zeros(size(b2));


%{
for i=1:m,
  x=sum(a1(i,:)/m);
  delta3 = -(y(:,i) - hx(:,i)) .* sigmoidGradient(z3(:,i)); 
  delta2 = (W2'*delta3 + beta*(-sparsityParam/x + (1-sparsityParam)/(1-x))) .* sigmoidGradient(z2(:,i));
  gradW2 = gradW2 + delta3*a2(:,i)';
  gradW1 = gradW1 + delta2*a1(:,i)'; 
  gradb1 = gradb1 + delta2;
  gradb2 = gradb2 + delta3;
end
%}


sparsity_delta = - sparsityParam ./ rho + (1 - sparsityParam) ./ (1 - rho);

  delta3 = -(y - hx) .* sigmoidGradient(z3); %64 10000
  delta2 = (W2'*delta3 + beta*(sparsity_delta*weightsbuffer)) .* sigmoidGradient(z2); %hidden size 10000
  gradW2 = gradW2 + delta3*a2';
  gradW1 = gradW1 + delta2*a1'; 
  gradb1 = gradb1 + sum(delta2')';
  gradb2 = gradb2 + sum(delta3')';

W1grad=gradW1/m +lambda*W1; 
b1grad=gradb1/m;
b2grad=gradb2/m;
W2grad=gradW2/m +lambda*W2;













%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

