function [T,Terstr,Rj] = sample_prune(T,Terstr,j,p,t,m,trainx,ytilta,mu,ntrain, pprune, pgrow, sigma2, sigmamu2,alpha,beta)     
    Tterminal = T(j).Terminal;    

    % uniformly selecting which terminal node to prune on 
    temp = Tterminal(rem(Tterminal,2)==0);
    parent = temp(ismember(Tterminal(rem(Tterminal,2)==0)+1,Tterminal))/2;
    changeprob = (1/length(parent):1/length(parent):1);
    parentindex = parent(find(changeprob>rand,1));

    % how deep is the parent of the selected terminal?
    a = parentindex;
    detastar = 0;
    while a ~= 1
        detastar = detastar+1;
        a = floor(a/2);
    end
    
    % Calculate the M-H ratio
    % 1. Calculate all the needed numbers
    w2 = length(parent); % number of singly internal parent nodes which have two terminal children
    
    leftind = 2*parentindex; rightind = 2*parentindex+1;
    childindex = [leftind,rightind];
    fieldname = 't'+string(childindex(1));
    subindex = Terstr(j).(fieldname);
    fieldname = 't'+string(childindex(2));
    subindex = [subindex; Terstr(j).(fieldname)]; % this is the index for the data in the parent node
    
    xgrow = trainx(subindex,:); 
    tmp = T(j).spvar; 
    splitvark = tmp(ismember(T(j).Internal,parentindex));
    [~,ia,~]=unique(xgrow(:,splitvark)); netastar = length(ia);

    % 2. find the sub-dataset in the parent node  
    Rj = computeRj(T, Terstr,m,mu,j,ytilta,ntrain);
    Rl = Rj(subindex); nl = length(Rl);
    fieldleft = 't'+string(leftind); fieldright = 't'+string(rightind);
    leftdata = Terstr(j).(fieldleft); rightdata = Terstr(j).(fieldright);
    RL = Rj(leftdata); nL = length(RL);
    RR = Rj(rightdata); nR = length(RR);
    
    % 3. Calculate the M-H ratio
    logAp = log(pgrow/pprune)+log(w2)-log((t-1)*p*netastar);
    logBp = -((1/2)*(log(sigma2*(sigma2+nl*sigmamu2))-log((sigma2+nL*sigmamu2)*(sigma2+nR*sigmamu2)))+...
        sigmamu2/(2*sigma2)*((sum(RL))^2/(sigma2+nL*sigmamu2)+(sum(RR))^2/(sigma2+nR*sigmamu2)-(sum(Rl))^2/(sigma2+nl*sigmamu2)));
    logCp = -(log(alpha)+2*log(1-alpha/(2+detastar)^beta)-log((1+detastar)^beta-alpha)-log(p*netastar));
    logrp = logAp+logBp+logCp;
    
    % Accept or not?
    up = rand;
    if log(up) <= logrp % accept
        % delete the orignal terminal nodes
        T(j).Terminal(ismember(T(j).Terminal, childindex)) = [];
        
        % add the parent node as a new terminal node
        T(j).Terminal = sort([T(j).Terminal, parentindex]);
        
        % delete the corresponding splitting variables and rules
        T(j).spvar(ismember(T(j).Internal,parentindex)) = [];
        T(j).sprule(ismember(T(j).Internal,parentindex)) = [];
        
        % delete the internal node
        T(j).Internal(ismember(T(j).Internal,parentindex)) = [];
        
        % add the new subdataset in Terstr
        fieldname = 't'+string(parentindex);
        fieldchild1 = 't'+string(childindex(1)); fieldchild2 = 't'+string(childindex(2));
        Terstr(j).(fieldname) = [Terstr(j).(fieldchild1); Terstr(j).(fieldchild2)];
        
        % delete the index in Terstr
        Terstr(j).(fieldchild1) = []; 
        Terstr(j).(fieldchild2) = []; 
    end
end