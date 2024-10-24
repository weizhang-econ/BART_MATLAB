function Rj = computeRj(T, Terstr,m,mu,j,ytilta,ntrain)
ghatnotj = zeros(ntrain,m);
for jj = 1:m
    if jj ~= j
        ghatnotj(:,jj) = computeghatj(T, Terstr,jj,mu,ntrain);
    end
end
Rj = ytilta - sum(ghatnotj,2);
end