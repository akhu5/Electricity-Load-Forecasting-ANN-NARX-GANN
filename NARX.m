% elctctricity load forecasting problem with external Input with a NARX Neural Network
i=xlsread('inputdata.xlsx');
t=xlsread('targetdata.xlsx');
%p = xlsread("predict.xlsx");
X = tonndata(i,false,false);
T = tonndata(t,false,false);
 
%Training Function
trainFcn = 'trainbr';  % Bayesian Regularization backpropagation.
 
%Nonlinear Autoregressive Network with External Input
inputDelays = 1:2;
feedbackDelays = 1:2;
hiddenLayerSize = 25;
 
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);
 
% Prepare of Data for Training and Simulation
% The function PREPARETS prepares timeseries data for a particular network,
% shifting time by the minimum amount to fill input states and layer
% states. Using PREPARETS allows us to keep our original time series 
[x,xi,ai,t] = preparets(net,X,{},T);
 
% Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 10/100;
net.divideParam.mu=.0001;
net.trainParam.epochs=1000 ;
net.trainParam.goal=1e-25;
net.trainParam.lr=.01;
 
% Train the Network
[net,tr] = train(net,x,t,xi,ai);
 
% Test the Network
y = net(x,xi,ai);
e = gsubtract(t,y);
performance = perform(net,t,y)
 
% View the Network
view(net)
%weight and bias
b1=net.b{1,1};
    w1=net.IW{1,1};
    %weigt and bais to output layer
    b2=net.b{2,1};
    w2=net.LW{2,1};
 
% Plots
 
 plotperform(tr)
 plottrainstate(tr)
 ploterrhist(e)
 plotregression(t,y)
 plotresponse(t,y)
 ploterrcorr(e)
 plotinerrcorr(x,e)
 
% Closed Loop Network
% Use this network to do multi-step prediction.
% The function CLOSELOOP replaces the feedback input with a direct
% connection from the output layer.
netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];
view(netc)
[xc,xic,aic,tc] = preparets(netc,X,{},T);
yc = netc(xc,xic,aic);
closedLoopPerformance = perform(net,tc,yc)
 
% Step-Ahead Prediction Network
nets = removedelay(net);
nets.name = [net.name ' - Predict One Step Ahead'];
view(nets)
[xs,xis,ais,ts] = preparets(nets,X,{},T);
ys = nets(xs,xis,ais);
stepAheadPerformance = perform(nets,ts,ys)