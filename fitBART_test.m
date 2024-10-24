function [ytilta,yhat] = fitBART_test(TREES, testx, miny, maxy,p,m)
nsims = size(TREES,2); [n,ptest] = size(testx);
if ptest ~= p
    disp(['Dimension of test set not correct.']);
end

%testy = testset(:,1);
%testytilta = -0.5+(testy - miny)/(maxy-miny);
%testset(:,1) = testytilta;
%testx = testset(:,2:end);
ghat = zeros(nsims,n,m);
for k = 1:nsims
    %disp([ num2str(k-nsims) ' more test loops to go...']);
    mu = TREES(k).mu;
    T = TREES(k).Tree;
    for j = 1:m
        testind = (1:1:n); tmpx = testx;
        Terminal = T(j).Terminal; Mu = mu(j).mu;
        [Internal,intindex] = sort(T(j).Internal);
        spvar = T(j).spvar(intindex); sprule = T(j).sprule(intindex);
        Intlength = length(Internal);
        for intl = 1:Intlength
            spnode = Internal(intl);
            if rem(spnode,2)==0
                tmpx = testx(RLind,:); testind = RLind;
            elseif rem(spnode,2)~=0 && spnode ~= 1
                tmpx = testx(RRind,:); testind = RRind;
            end
            spvartmp = spvar(intl); spruletmp =sprule(intl);
            RLind = testind(tmpx(:,spvartmp) <= spruletmp);
            RRind = testind(tmpx(:,spvartmp) > spruletmp);
            childl = 2*spnode;
            childr = 2*spnode+1;
            if ismember(childl, Terminal)
                ghat(k,RLind,j) = Mu(Terminal == childl);
            end
            if ismember(childr, Terminal)
                ghat(k,RRind,j) = Mu(Terminal == childr);
            end
        end
    end
end
ytilta = squeeze(sum(ghat,3));
ytiltahat = mean(squeeze(sum(ghat,3)))';
yhat = (ytiltahat+0.5)*(maxy-miny)+miny;
%rmse = sqrt(mean((testy-yhat).^2));
end