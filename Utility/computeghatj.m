function ghatj = computeghatj(T, Terstr,j,mu,ntrain)
    Terminal = T(j).Terminal; t = length(Terminal);
    ghatj = zeros(ntrain,1);
    for tt = 1:t
        %fieldname = ['t',num2str(Terminal(tt))];
        %fieldname = ['t',sprintf('%d',Terminal(tt))];
        fieldname = 't'+string(Terminal(tt));
        terindex = Terstr(j).(fieldname);
        mus = mu(j).mu;
        ghatj(terindex) = mus(tt);
    end
end
