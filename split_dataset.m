function [X_train, y_train, X_test, y_test] = split_dataset(X, y, percent)
  % X -> the loaded dataset with all training examples
  % y -> the corresponding labels
  % percent -> fraction of training examples to be put in training dataset

  % X_[train|test] -> the datasets for training and test respectively
  % y_[train|test] -> the corresponding labels

  % Example: [X, y] has 1000 training examples with labels and percent = 0.85
  %           -> X_train will have 850 examples
  %           -> X_test will have the other 150 examples

 % Split a dataset into training and test sets based on a percentage split

  % Shuffle the examples randomly
  idx = randperm(size(X, 1));
  X = X(idx, :);
  y = y(idx);

  % Determine the number of examples in the training set based on the percentage split
  num_train = round(size(X, 1) * percent);

  % Extract the training and test sets from the shuffled data
  X_train = X(1:num_train, :);
  y_train = y(1:num_train);
  X_test = X(num_train+1:end, :);
  y_test = y(num_train+1:end);
endfunction
