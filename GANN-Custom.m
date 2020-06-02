% INITIALIZE THE NEURAL NETWORK PROBLEM %
% inputs for the neural net
inputs = xlsread("inputdata.xlsx");
% targets for the neural net
targets = xlsread("targetdata.xlsx");
inputs=inputs';
targets=targets';
% number of neurons
n = 2;
% create a neural network
net = feedforwardnet(n);
% configure the neural network for this dataset
net = configure(net, inputs, targets);
x = getwb(net);
% create handle to the MSE_TEST function, that
% calculates MSE
h = @(x) mse_test(x, net, inputs, targets);
% Setting the Genetic Algorithms tolerance for
% minimum change in fitness function before
% terminating algorithm to 1e-8 and displaying
% each iteration's results.
ga_opts = gaoptimset('TolFun', 1e-1,'display','iter');
% PLEASE NOTE: For a feed-forward network
% with n neurons, 3n+1 quantities are required
% in the weights and biases column vector.
%
% a. n for the input weights
% b. n for the input biases
% c. n for the output weights
% d. 1 for the output bias
% running the genetic algorithm with desired options
[x_ga_opt, err_ga] = ga(h, 43, ga_opts);
function mse_calc = mse_test(x, net, inputs, targets)
% 'x' contains the weights and biases vector
% in row vector form as passed to it by the
% genetic algorithm. This must be transposed
% when being set as the weights and biases
% vector for the network.
% To set the weights and biases vector to the
% one given as input
%wb = formwb(net,net.b,net.IW,net.LW);
net = setwb(net, x');
% To evaluate the ouputs based on the given
% weights and biases vector
y = net(inputs);
% Calculating the mean squared error
mse_calc = sum((y-targets).^2)/length(y);
end