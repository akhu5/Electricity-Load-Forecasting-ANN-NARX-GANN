function [Y,Xf,Af] = myNeuralNetworkFunction(X,Xi,~)
%MYNEURALNETWORKFUNCTION neural network simulation function.
%
% Generated by Neural Network Toolbox function genFunction, 25-Mar-2019 14:36:43.
%
% [Y,Xf,Af] = myNeuralNetworkFunction(X,Xi,~) takes these arguments:
%
%   X = 2xTS cell, 2 inputs over TS timesteps
%   Each X{1,ts} = 4xQ matrix, input #1 at timestep ts.
%   Each X{2,ts} = 1xQ matrix, input #2 at timestep ts.
%
%   Xi = 2x2 cell 2, initial 2 input delay states.
%   Each Xi{1,ts} = 4xQ matrix, initial states for input #1.
%   Each Xi{2,ts} = 1xQ matrix, initial states for input #2.
%
%   Ai = 2x0 cell 2, initial 2 layer delay states.
%   Each Ai{1,ts} = 10xQ matrix, initial states for layer #1.
%   Each Ai{2,ts} = 1xQ matrix, initial states for layer #2.
%
% and returns:
%   Y = 1xTS cell of 2 outputs over TS timesteps.
%   Each Y{1,ts} = 1xQ matrix, output #1 at timestep ts.
%
%   Xf = 2x2 cell 2, final 2 input delay states.
%   Each Xf{1,ts} = 4xQ matrix, final states for input #1.
%   Each Xf{2,ts} = 1xQ matrix, final states for input #2.
%
%   Af = 2x0 cell 2, final 0 layer delay states.
%   Each Af{1ts} = 10xQ matrix, final states for layer #1.
%   Each Af{2ts} = 1xQ matrix, final states for layer #2.
%
% where Q is number of samples (or series) and TS is the number of timesteps.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [1;17;4;736787];
x1_step1.gain = [0.0869565217391304;0.0909090909090909;0.125;0.25];
x1_step1.ymin = -1;

% Input 2
x2_step1.xoffset = 2443.71;
x2_step1.gain = 0.00114707839133726;
x2_step1.ymin = -1;

% Layer 1
b1 = [-0.94396292513482494;-0.1310021726648703;-0.65706147723575337;0.49410464149871086;-0.17089825056860916;0.17751058424794108;0.84015079597515463;-0.3629046489459099;0.053380101884830958;0.46715329284523599];
IW1_1 = [0.30806622143730927 0.081848919100430309 0.14392287670571183 0.54734678094542877 0.57728598467052816 -0.2594329254366613 -0.22512141460740062 0.52492944394493846;0.10882309022417813 -0.33148854889515733 0.24320727910376627 -1.0679544491611592 0.015036187040022148 0.29852308062881538 0.4733311029756635 -1.055354953751964;-0.12628985631723599 -0.46828695226855094 -0.30340831665924411 -0.0091702325732905769 0.2128957958184837 0.45027082547899394 0.12793884164166966 -0.042957456764178253;-0.040351785468319731 0.41850645951832521 0.058590545687843373 0.12163666233531721 -0.065253580284213672 0.52994823534407898 -0.34822331866233508 0.11947276649035905;-0.25196158544526603 -0.28330133089844639 0.042293958170726897 -0.039841667408885832 -0.18617030686876174 -0.056468179860315833 -0.3095251606645637 -0.045942724211607069;0.73817181836360257 0.61066478881327968 -0.29133888000736013 -0.068962372783143003 0.7642300549585973 0.23325642814104078 0.070707632325972145 -0.07393300096116856;-0.24369197314461002 0.29271625592376566 0.12339364839055747 0.78938720612917135 -1.1511211797570642 0.20998029026515602 -0.37950512956142157 0.88933809237987904;-0.38849163092676309 -0.0131205720088687 0.56162183332147086 0.24454859193022621 -0.070226527268379857 -0.1111918820036151 0.56142811456601893 0.21020334148100264;-0.406142236384056 -0.15379463780268315 0.18513212621906552 0.94962546329482678 -0.29620352312789439 0.53596073823858692 0.12812266378101611 0.93589967887471681;-0.47286658429219619 -0.068723557822451403 -0.62709319437891908 -0.21723658969137821 -0.72411958574300295 -0.19947189790954425 -0.57618045717741095 -0.19200475402638692];
IW1_2 = [0.21724298955940094 0.59985170305486646;0.35949856089296855 -0.54597861182877405;0.37145314583358741 0.67827076481022175;0.46721356290198113 0.56929084777454442;0.52134493109082547 -0.36607815959236167;-0.065645912934455791 -0.79321379302625761;-0.2456206208488752 -0.63912512350282558;0.16304757932175293 -0.17321308812786998;0.31508520098541032 -0.07634839170963921;-0.34372673129197834 -0.5037495820125969];

% Layer 2
b2 = -0.23822751702281336;
LW2_1 = [-1.1953845138931067 -0.76092299741455593 0.94584126841731031 0.48357368284703073 0.78859834351521729 0.92787357046303864 1.2190869664909636 0.88317418906142453 -1.3814767982673777 -0.8427542950859821];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = 0.00114707839133726;
y1_step1.xoffset = 2443.71;

% ===== SIMULATION ========

% Format Input Arguments
isCellX = iscell(X);
if ~isCellX
    X = {X};
end
if (nargin < 2), error('Initial input states Xi argument needed.'); end

% Dimensions
TS = size(X,2); % timesteps
if ~isempty(X)
    Q = size(X{1},2); % samples/series
elseif ~isempty(Xi)
    Q = size(Xi{1},2);
else
    Q = 0;
end

% Input 1 Delay States
Xd1 = cell(1,3);
for ts=1:2
    Xd1{ts} = mapminmax_apply(Xi{1,ts},x1_step1);
end

% Input 2 Delay States
Xd2 = cell(1,3);
for ts=1:2
    Xd2{ts} = mapminmax_apply(Xi{2,ts},x2_step1);
end

% Allocate Outputs
Y = cell(1,TS);

% Time loop
for ts=1:TS
    
    % Rotating delay state position
    xdts = mod(ts+1,3)+1;
    
    % Input 1
    Xd1{xdts} = mapminmax_apply(X{1,ts},x1_step1);
    
    % Input 2
    Xd2{xdts} = mapminmax_apply(X{2,ts},x2_step1);
    
    % Layer 1
    tapdelay1 = cat(1,Xd1{mod(xdts-[1 2]-1,3)+1});
    tapdelay2 = cat(1,Xd2{mod(xdts-[1 2]-1,3)+1});
    a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*tapdelay1 + IW1_2*tapdelay2);
    
    % Layer 2
    a2 = repmat(b2,1,Q) + LW2_1*a1;
    
    % Output 1
    Y{1,ts} = mapminmax_reverse(a2,y1_step1);
end

% Final Delay States
finalxts = TS+(1: 2);
xits = finalxts(finalxts<=2);
xts = finalxts(finalxts>2)-2;
Xf = [Xi(:,xits) X(:,xts)];
Af = cell(2,0);

% Format Output Arguments
if ~isCellX
    Y = cell2mat(Y);
end
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
y = bsxfun(@minus,x,settings.xoffset);
y = bsxfun(@times,y,settings.gain);
y = bsxfun(@plus,y,settings.ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
x = bsxfun(@minus,y,settings.ymin)
x = bsxfun(@rdivide,x,settings.gain)
x = bsxfun(@plus,x,settings.xoffset)
end
