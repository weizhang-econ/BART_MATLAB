%% BART %%
function [TREES,ytiltahat,trainyhat,ytiltatest,yhattest,trainrmse,miny,maxy,p] = BART(trainx, trainy, alpha, beta,m,nu,q,pgrow,pprune,iter,burn,testx)
[ntrain,p] = size(trainx); dataind = (1:1:ntrain)';
%% Find the index for binary variables
bivar = zeros(p,1);
for ip = 1:p
    tmp = unique(trainx(:,ip));
    if length(tmp) == 2
        bivar(ip) = 1;
    end
end

%% More hyperparameters to calculate
% sigmamu and mumu
miny = min(trainy);
maxy = max(trainy);
mumu = 0;
sigmamu = 0.5/(nu*sqrt(m));
sigmamu2 = sigmamu^2;

%lambda:OLS
lm = fitlm(trainx, trainy);
residuals = table2array(lm.Residuals);
residuals = residuals(:,1);
sigma2 = residuals'*residuals/(ntrain-p);
lambdatemp = @(lambda) lambdafxn(nu,q,sigma2,lambda);
lambda = fminsearch(lambdatemp,1);

% % lambda: naive specification
% sigma2 = var(trainy);
% lambdatemp = @(lambda) lambdafxn(nu,q,sigma2,lambda);
% lambda = fminsearch(lambdatemp,1);

%% Transform y into ytilta \in [-0.5, 0.5]
ytilta = -0.5+(trainy - miny)/(maxy-miny);
%trainset = [ytilta, trainx];

%% Initialize the tree T and corresponding mu (the associated value to each terminal node)
field1 = 'Terminal'; field2 = 'Internal'; field3 = 'spvar'; field4 = 'sprule'; 
value1 = num2cell(ones(1,m)); value2 = num2cell(NaN(1,m)); value3 = num2cell(NaN(1,m)); value4 = num2cell(NaN(1,m));
T = struct(field1,value1,field2,value2,field3,value3,field4,value4);

field1 = 't1'; value1 = num2cell(ones(1,m)); field2 = 't2'; value2 = num2cell(0);
Terstr = struct(field1,value1,field2,value2); 
for ind = 1:m
    Terstr(ind).(field1) = dataind;
end

field1 = 'mu'; value1 = num2cell(mean(ytilta)/m*ones(1,m));
mu = struct(field1,value1);

% Initialize the storage structure
field1 = 'Tree'; field2 = 'Terstr';field3 = 'mu'; field4 = 'variance'; 
value1 = num2cell(ones(1,iter)); value2 = num2cell(NaN(1,iter));value3 = num2cell(NaN(1,iter)); value4 = num2cell(NaN(1,iter));
Meta = struct(field1,value1,field2,value2,field3,value3,field4,value4);

rng('default')
tic;
for i = 1:iter+burn
    if rem(i,1000)==0
        disp([ num2str(i-(iter+burn)) ' more loops to go...']);      
    end
    
    for j = 1:m
        %% sample m trees one by one
        t = length(T(j).Terminal);
        if t == 1 % When there is only one node, trees can only be proposed to grow
            % sample the tree using the function "sample_oneternode"
            [T,Terstr, Rj] = sample_oneternode(T,Terstr,j,p,t,m,trainx,ytilta,mu,ntrain, pprune, pgrow, sigma2, dataind,sigmamu2,alpha,beta,bivar);
        else % when there are more than one node, trees can be proposed to grow, prune or change
            u2 = rand;
            if u2 <= 1/3 %grow
                [T,Terstr,Rj] = sample_grow(T,Terstr,j,p,t,m,trainx,ytilta,mu,ntrain, pprune, pgrow, sigma2,sigmamu2,alpha,beta);
                
            elseif u2 > 1/3 && u2 <= 2/3 %prune
                [T,Terstr,Rj] = sample_prune(T,Terstr,j,p,t,m,trainx,ytilta,mu,ntrain, pprune, pgrow, sigma2, sigmamu2,alpha,beta);
                
            else %change
                [T,Terstr,Rj] = sample_change(T,Terstr,j,p,m,trainx,ytilta,mu,ntrain,  sigma2, sigmamu2);
            
            end
            
        end
        
        %% sample muij from normal distribution
        B = length(T(j).Terminal); % # of terminals on tree j, this is the number of muij
        tmp = zeros(B,1);
        for b = 1:B
            % which node is it?
            index = T(j).Terminal(b);
            
%             % Calculate Rj
%             Rj = computeRj(T, Terstr,m,mu,j,ytilta,ntrain);
            
            % Decide which Rj are on which terminal node
            fieldname = strcat('t',num2str(index));
            Rbj = Rj(Terstr(j).(fieldname));
            nb = length(Rbj);
            
            % Calculate the mean
            meanmu = (sigmamu2*sum(Rbj)+sigma2*mumu)/(nb*sigmamu2+sigma2); % mean of the posterior distribution of muij
            
            % Calculate the variance
            variancemu = sigma2*sigmamu2/(nb*sigmamu2+sigma2); % variance of the posterior distribution of muij
            if variancemu < 0
                disp('Variance of muij is negative');
            end
            
            % Sample and store mu
            tmp(b) = meanmu+sqrt(variancemu)*randn;
            
        end
            mu(j).mu = tmp;
    end
    
    % sample sigma2 from inverse gamma
    abar = (nu+ntrain)/2;
    ghat = computeghat(T, Terstr,m,mu,ntrain);
    temp = sum((ytilta - ghat).^2);
    bbar = 1/2*(nu*lambda+temp);
    sigma2 = 1./gamrnd(abar,1/bbar);
    
    % store the structure
    if i > burn
        Meta(i-burn).Tree = T;
        Meta(i-burn).Terstr = Terstr;
        Meta(i-burn).mu = mu;
        Meta(i-burn).variance = sigma2;
    end
end
toc;
%% Evaluation
% 1. Convergence
%variancestore = vertcat(Meta.variance);
%figure;
%plot(variancestore);
[ytiltahat,trainyhat] = fitBART(Meta, ntrain, miny, maxy,m);
% figure;
% for i = 1:ntrain
%     plot(1:nsims,trainyhat(:,i))
%     pause;
% end

% 2. Fitness of the model: train set
trainyhatv = mean(trainyhat)';
trainrmse = sqrt(mean((trainy-trainyhatv).^2));

% % 3. Fitness of the model: test set
 TREES = rmfield(Meta,'Terstr');
 [ytiltatest,yhattest] = fitBART_test(TREES, testx, miny, maxy,p,m);
end
