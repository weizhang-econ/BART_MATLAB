function RMSEblmtest = BLM(iter,burn,beta0,iVbeta0,nu0,S0, trainx, trainy, testset)
[ntrain, p] = size(trainx);
% initialization
beta = (trainx'*trainx)\(trainx'*trainy);
sig2 = sum((trainy-trainx*beta).^2)/(ntrain-p);
store_theta = zeros(iter-burn,p+1);

for i = 1:iter
    % sample beta
    Dbeta = (iVbeta0+trainx'*trainx/sig2)\speye(p);
    beta_hat = Dbeta*(iVbeta0*beta0+trainx'*trainy/sig2);
    C = chol(Dbeta,'lower');
    beta = beta_hat+C*randn(p,1);
    
    % sample sig2
    e =trainy - trainx*beta;
    sig2 = 1/gamrnd(nu0+ntrain/2,1/(S0+e'*e/2));
    
    % store the parameters
    if i > burn
        store_theta(i-burn,:)=[beta', sig2];
    end
end

% calculate the RMSEtest
testy = testset(:,1);
testx = [ones(size(testset,1),1),testset(:,2:end)];
beta = mean(store_theta(:,1:p));
testyhat = testx*beta';
RMSEblmtest = sqrt(mean((testy - testyhat).^2));
end