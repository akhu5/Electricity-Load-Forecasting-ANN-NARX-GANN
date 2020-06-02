% Neural Network trained by Genetic Algorithm (GA)
% Reference: http://www.mathworks.com/matlabcentral/answers/100323
rng('default')
[x, t] = simplefit_dataset;
[I, ~] = size(x);
[O, N] = size(t);
% Reference MSE: Average Target Variance
var_t = mean(var(t,1,2));
% Hidden Node Choice
H = 4;
Nw = (I + 1)*H + (H + 1)*O;
% Create Regression/Curve-Fitting Neural Network:
net = feedforwardnet(H);
net.divideFcn = 'dividetrain';
% Configure the Net for the Simplefit Dataset
net = configure(net, x, t);
% Initial Weights and Errors
wb = getwb(net)';
% Create handle to the error function,
fun = @(wb) nmse(wb, net, x, t); % NMSE
% Set the Genetic Algorithm tolerance for minimum change in fitness function
% before terminating algorithm to 1e-4 and display each iteration's results.
opts = optimoptions('ga', 'FunctionTolerance', 1e-4, 'Display', 'iter');
tic
[wbopt, fval] = ga(fun, Nw, opts);
totaltime = toc;
% Assign the weights to net
net = setwb(net, wbopt');
% Simulate the output
y = sim(net,x);

function nmse_calc = nmse(wb, net, input, target)
      % 'wb' contains the weights and biases vector
      % in row vector form as passed to it by the
      % genetic algorithm. This must be transposed
      % when being set as the weights and biases
      % vector for the network.
      % Reference MSE
      var_t = mean(var(target,1,2));
      % To set the weights and biases vector to the
      % one given as input
      net = setwb(net, wb');
      % To evaluate the ouputs based on the given
      % weights and biases vector
      y = net(input);
      % Calculating the Normalised Mean Squared Error (NMSE)
      nmse_calc = mean((target(:) - y(:)).^2) / var_t;
end