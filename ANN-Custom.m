ip1 = xlsread("inputdata.xlsx");
tar = xlsread("targetdata.xlsx"); 
test1 = xlsread("testingdata.xlsx");
ip=transpose(ip1);
test=transpose(test1);
%[ip,test] = simplefit_dataset;
% Create a Fitting Network
hiddenLayerSize = 20; 
net = fitnet(hiddenLayerSize);
view(net);
% Setup Division of Data for Training, Validation, Testing 
%net.divideParam.trainRatio = 70/100;
%net.divideParam.valRatio = 15/100;
%net.divideParam.testRatio = 15/100;
net.trainparam.epochs= 195;
net.trainparam.goal= 1e-5;
net.trainparam.lr= 0.01;
net = train(net,ip,test);
view(net)
ypred1=sim(net,tar')';
xlswrite('output.xlsx',ypred1);