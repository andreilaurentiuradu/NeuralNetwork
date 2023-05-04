function [classes] = predict_classes(X, weights, ...
                  input_layer_size, hidden_layer_size, ...
                  output_layer_size)
  % X -> the test examples for which the classes must be predicted
  % weights -> the trained weights (after optimization)
  % [input|hidden|output]_layer_size -> the sizes of the three layers

  % classes -> a vector with labels from 1 to 10 corresponding to
  %            the test examples given as parameter

  % TODO: predict_classes implementation

  % Reshape parameters
  Theta1_unrolled = weights(1:((input_layer_size + 1) * hidden_layer_size));
  Theta2_unrolled = weights((((input_layer_size + 1) * hidden_layer_size) + 1):end);

  Theta1 = reshape(Theta1_unrolled, [hidden_layer_size, input_layer_size + 1]);
  Theta2 = reshape(Theta2_unrolled, [output_layer_size, hidden_layer_size + 1]);

  % Add bias term to input layer
  X = [ones(size(X,1),1) X];
  a1 = X;

  % Compute hidden layer activations
  z2 = X * Theta1';
  a2 = sigmoid(z2);

  % Add bias term to hidden layer
  a2 = [ones(size(a2,1),1) a2];

  % Compute output layer activations
  activations = sigmoid(a2 * Theta2');

  % Get the largest activation
  [max_value, max_index] = max(activations, [], 2);

  classes = max_index;
endfunction
