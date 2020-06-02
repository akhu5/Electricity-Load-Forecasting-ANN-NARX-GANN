 ip = xlsread("inputdata.xlsx");
 targets = xlsread("targetdata.xlsx"); 
p = xlsread("predict.xlsx"); 
%tr=transpose(tr);
ip=transpose(ip);
inputs = ip;
 targets =targets'; 
% Create a Fitting Network
 hid = 20; 
net = fitnet(hid,'trainbr'); 
% Setup Division of Data for Training, Validation, Testing 
net.divideParam.trainRatio = 80/100;
 net.divideParam.valRatio = 10/100;
 net.divideParam.testRatio = 10/100;
 %net.trainParam.mu = 0.0001;
 net.trainparam.epochs= 1990;
net.trainparam.goal= 1e-25;
%net.trainparam.lr= 0.01;
% Train the Network:
for i=1:5
 [net,tr] = train(net,inputs,targets);
end
 % Test the Network 
outputs = net(inputs);
 errors = gsubtract(targets,outputs); 
performance = perform(net,targets,outputs);
% View the Network
 view(net);
 a=sim(net,p') 
%a1=sim(net,ip)';

% weight and bias to hidden layer 
b1=net.b{1,1}; 
W1=net.IW{1,1}; 
% weight and bias to output layer 
b2 = net.b{2,1};
 W2 = net.LW{2,1};
% Plots 
plotperform(tr);
plottrainstate(tr); 
plotfit(net,inputs,targets); 
plotregression(targets,outputs); 
ploterrhist(errors);