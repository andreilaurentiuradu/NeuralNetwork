function [X, y] = load_dataset(path)
  % path -> a relative path to the .mat file that must be loaded

  % X, y -> the training examples (X) and their corresponding labels (y)

  % Load a dataset from a .mat file at the given path

  % Load the .mat file into a struct
  data = load(path);

  % Extract the feature matrix X and label vector y from the struct
  X = data.X;
  y = data.y;
endfunction
