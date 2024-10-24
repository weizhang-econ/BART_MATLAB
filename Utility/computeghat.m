function ghat = computeghat(T, Terstr,m,mu,ntrain)
tmp = 0;
for j = 1:m
    Terminal = T(j).Terminal; t = length(Terminal);
    ghatj = zeros(ntrain,1);
    for tt = 1:t
        fieldname = strcat('t',num2str(Terminal(tt)));
        terindex = Terstr(j).(fieldname);
        mus = mu(j).mu;
        ghatj(terindex) = mus(tt);
    end
    tmp = tmp+ghatj;
end
ghat = tmp;
end