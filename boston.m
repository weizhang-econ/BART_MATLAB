%% Analyze Boston Housing using BART %%
%% Wei Zhang %%
%% Dec. 2020 %%

% Load the data
clear;
restoredefaultpath;
addpath('.\Utility')
filename = 'BostonHousing.csv';
BostonHousing = table2array(readtable(filename));

% Process the data
dataset = [BostonHousing(:,end),BostonHousing(:,1:end-1)];%the last variable is the price of the house, thus our y.

[n,xandy] = size(dataset);
p = xandy-1; % number of explanatory variables
%% Set the hyperparameters: alpha, beta,mumu,sigmamu,nu,lambda
alpha = 0.95;
beta = 2;
m = 5;
v = 2;
nu = 3;
q = 0.9;
pgrow = 1/3;
pprune = 1/3;
pchange = 1/3;

%% Prior for Bayesian Linear Model
beta0 = zeros(p+1,1);
iVbeta0 = eye(p+1)/100;
nu0 = 3;
S0 = 1*(nu0-1);

%% Iterations
iter = 4000;
burn = 1000;

%% Cross Validation
RMSEtest = zeros(10,1);
RMSElm = zeros(10,1);
RMSEblm = zeros(10,1);
rng('default')
tic;
for i = 1:10
    % Partition the dataset
    partition = cvpartition(n,'Holdout',0.2);    
    idxTrain = training(partition);
    idxTest = test(partition);
   
    % BART 
    trainset = dataset(idxTrain,:);
    trainx = trainset(:,2:p+1);
    trainy = trainset(:,1);
    testx = dataset(idxTest,2:p+1);
    testy = dataset(idxTest,1);                                          
    [TREES,ytiltahat,trainyhat,ytiltatest,yhattest,trainrmse,miny,maxy,p] = BART(trainx, trainy, alpha, beta,m,nu,q,pgrow,pprune,iter,burn,testx);
    [ytilta,yhat] = fitBART_test(TREES, testx, miny, maxy,p,m);
    RMSEtest(i) = sqrt(mean((testy-yhat).^2));
    
    % Linear Regression
    lm = fitlm(trainx,trainy);
    testx = [ones(size(testx,1),1),dataset(idxTest,2:p+1)];
    testset = dataset(idxTest,:);
    testyhat = testx*table2array(lm.Coefficients(:,1));
    RMSElm(i) = sqrt(mean((testy-testyhat).^2));
    
    % Bayesian Linear Regression
    trainx = [ones(size(trainset,1),1),trainset(:,2:p+1)];
    RMSEblm(i) = BLM(iter,burn,beta0,iVbeta0,nu0,S0, trainx, trainy, testset);  
end
toc;
mean(RMSEtest)
mean(RMSElm)
mean(RMSEblm)
