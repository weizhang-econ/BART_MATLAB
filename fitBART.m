%% this function find the yhat for train data set based on the structure Terstr
function [ytiltahat,yhat] = fitBART(Meta, ntrain, miny, maxy,m)
nsims = size(Meta,2); ghat = zeros(nsims,ntrain,m);
parfor k = 1:nsims
    Terstr = Meta(k).Terstr;
    mu = Meta(k).mu;
    T = Meta(k).Tree;
    for j = 1:m
        ghat(k,:,j) = computeghatj(T, Terstr,j,mu,ntrain)';
    end
end
ytiltahat = squeeze(sum(ghat,3));
yhat = (ytiltahat+0.5)*(maxy-miny)+miny;
end
