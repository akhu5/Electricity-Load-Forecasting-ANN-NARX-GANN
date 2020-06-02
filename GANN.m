%for i = 1:3
inputs = xlsread('inputdata.xlsx')';
targets = xlsread('targetdata.xlsx')';
predict = xlsread('predict.xlsx')';
%c = xlsread('Compare.xlsx')';
%inputs = p(:,i+20:i+27);
%targets = t(:,i+20:i+27);
%in = IN(:,i);
%C = c(:,i);
[I N ] = size(inputs);
[O N ] = size(targets);
H = 1;
Nw = (I+1)*H+(H+1)*O;
net = feedforwardnet(H); 
net = configure(net, inputs, targets);
wb_in = getwb(net)';
h = @(x) mse_test(x, net, inputs, targets);
ga_opts=gaoptimset('TolFun',1e-20,'Display','iter','Generations',20,'PopulationSize',400,'MutationFcn',@mutationgaussian,'CrossoverFcn',@crossoverscattered,'UseParallel', true);
[x_ga_opt, err_ga] = ga(h, Nw,[],[],[],[],[],[],[], ga_opts);
net = setwb(net, x_ga_opt');
outputs = net(inputs);
errors = gsubtract(targets,outputs); 
performance = perform(net,targets,outputs);
wb_fin = getwb(net)';
predictoutput=sim(net,predict);
%Sheet = 1;
%filename = 'Results.xlsx';
%xlRange =['A',num2str(i)];
%xlswrite(filename,x_ga_opt,Sheet,xlRange);
%i = i + 1;
%end
%-------------------------------Objective Function---------------------------------
function mse_calc = mse_test(x, net, inputs, targets)
net = setwb(net, x');
y = net(inputs);
e = targets - y;
mse_calc = mse(e);
end