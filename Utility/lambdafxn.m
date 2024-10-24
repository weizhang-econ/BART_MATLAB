function [distance,lambda] = lambdafxn(nu,q,sigma2,lambda)
    a = nu/2;
    b = nu*lambda/2;
    p = 1 - gamcdf(1/sigma2,a,1/b);
    distance = abs(p - q);
end