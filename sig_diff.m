function y = sig_diff(x)
  y = sigmoid(x) .* (1 - sigmoid(x));
endfunction
