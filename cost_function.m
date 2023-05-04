function [J, grad] = cost_function(params, X, y, lambda, input_layer_size,
  hidden_layer_size, output_layer_size)

  % Reshape the parameter vector into the Theta1 and Theta2 matrices
  Theta1 = reshape(params(1:hidden_layer_size * (input_layer_size + 1)), [hidden_layer_size, input_layer_size + 1]);
  Theta2 = reshape(params((1 + (hidden_layer_size * (input_layer_size + 1))):end), [output_layer_size, hidden_layer_size + 1]);

  m = size(X, 1); % number of training examples

  % Add ones to the X data matrix
  X = [ones(m, 1) X];
  a1 = X;
  % Compute the hypothesis
  z2 = X * Theta1';
  a2 = sigmoid(z2);

  % bias
  a2 = [ones(size(a2,1),1) a2];
  activations = sigmoid(a2 * Theta2');

  a3 = activations;

  Y = zeros(numel(y), 10);
  for i = 1:numel(y)
    Y(i, y(i)) = 1;
  endfor
  m = size(a2, 1);






  J = sum(sum((-Y .* log(activations) - (1 - Y) .* log(1 - activations)))) / m;
  J += (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2))) * (lambda / (2 * m));
  disp(J);

  % -----BACKPROPAGATION----------------
  D1 = zeros(size(Theta1));
  D2 = zeros(size(Theta2));


  delta3 = a3 - Y;
  D2 = D2 + delta3' * a2;

  delta2 = (delta3 * Theta2) .* sig_diff([ones(m, 1) z2]);
  delta2 = delta2(:, 2:end);
  D1 = D1 + delta2' * a1;

  % Regularize gradient, without bias
  DJ_DTheta1 = (1/m) * D1 + (lambda/m) * [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)];
  DJ_DTheta2 = (1/m) * D2 + (lambda/m) * [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)];

  grad = [DJ_DTheta1(:); DJ_DTheta2(:)];

endfunction
