function [T,Terstr,Rj] = sample_grow(T,Terstr,j,p,t,m,trainx,ytilta,mu,ntrain, pprune, pgrow, sigma2,sigmamu2,alpha,beta)     
     % which node to grow on
     Tterminal = T(j).Terminal;
     probnode = (1/t:1/t:1); Ssplitind = find(probnode>rand,1);
     Ssplit = Tterminal(Ssplitind);
    
     % how deep is this node
     a = Ssplit;
     deta = 0;
     while a ~= 1
         deta = deta+1;
         a = floor(a/2);
     end
     
     % which splitting variable
     fieldname = 't'+string(Ssplit);
     termindex = Terstr(j).(fieldname);
     xgrow = trainx(termindex,:);
% if there is only one data in the terminal node, find a different
     % splitting node
     while size(xgrow,1) == 1 || size(unique(xgrow(:,1)),1) == 1
         Ssplitind = find(probnode>rand,1);Ssplit = Tterminal(Ssplitind);
         a = Ssplit;
         deta = 0;
         while a ~= 1
             deta = deta+1;
             a = floor(a/2);
         end
         fieldname = 't'+string(Ssplit);
         termindex = Terstr(j).(fieldname);
         xgrow = trainx(termindex,:);
     end
     varlist = find( arrayfun( @(c) numel( unique( xgrow(:, c) )), 1:p) >= 2 )';
     varnum = length(varlist);
     probx = (1/varnum:1/varnum:1);
     k = varlist(find(probx>rand,1));
     
     % which splitting rule to use
     [~,ia,~] = unique(xgrow(:,k));
     neta = length(ia);
     probvalue = (1/neta:1/neta:1);
     e = find(probvalue>rand,1);
     splitvalue = xgrow(ia(e),k);
     
     % Calculate M-H ratio
     %Tterminalbefore = Tterminal;
     newTterminal = [2*Ssplit, 2*Ssplit+1];
     Internal = [T(j).Internal,Ssplit];  % old terminal node become internal node
     oldterminal = Tterminal;
     Tterminal = Tterminal(Tterminal ~= Ssplit);
     Tterminal(end+1:end+length(newTterminal)) = newTterminal;
     Tterminal = sort(Tterminal);
     temp = Tterminal(rem(Tterminal,2)==0);
     w2_star = sum(ismember((temp+1),Tterminal));
     
     Rj = computeRj(T, Terstr,m,mu,j,ytilta,ntrain);
     fieldname = 't'+string(Ssplit);
     Terindex = Terstr(j).(fieldname);
     Rl = Rj(Terindex);
     nl = length(Rl);
     
     % which Rjs are going to be in the left and right node
    % here we need to make sure every terminal node has at least one
    % observation. In order to do this, we need to avoid the following
    % cases: (1) binary variables (2) splitting rule is the maximum or the
    % minimum of the selected variable.
    tmp = sort(unique(xgrow(:,k)));
    if tmp==2 % the case of binary variable
        RLind =termindex(xgrow(:,k) == tmp(1));
        RL = Rj(RLind);
        nL = length(RL);
        RRind = termindex(xgrow(:,k) == tmp(2));
        RR = Rj(RRind);
        nR = length(RR);
    elseif splitvalue == max(xgrow(:,k)) % the case where the maximum value of the splitting variable is selected
        RLind = termindex(xgrow(:,k)<splitvalue);
        RL = Rj(RLind);
        nL = length(RL);
        RRind = termindex(xgrow(:,k)==splitvalue);
        RR = Rj(RRind);
        nR = length(RR);
    else
        RLind = termindex(xgrow(:,k)<=splitvalue);
        RL = Rj(RLind);
        nL = length(RL);
        RRind = termindex(xgrow(:,k)>splitvalue);
        RR = Rj(RRind);
        nR = length(RR);
    end
    
    % 3. M-H ratio
    logA = log(pprune/pgrow)+log(t*p*neta)-log(w2_star);
    logB = (1/2)*(log(sigma2*(sigma2+nl*sigmamu2))-log((sigma2+nL*sigmamu2)*(sigma2+nR*sigmamu2)))+...
        sigmamu2/(2*sigma2)*((sum(RL))^2/(sigma2+nL*sigmamu2)+(sum(RR))^2/(sigma2+nR*sigmamu2)-(sum(Rl))^2/(sigma2+nl*sigmamu2));
    logC = log(alpha)+2*log(1-alpha/(2+deta)^beta)-log((1+deta)^beta-alpha)-log(p*neta);
    logr = logA+logB+logC;
    
    % Accept or not?
    ug2 = rand;
    if log(ug2) <= logr % Accept and grow on the selected splitting node with the splitting rule, renew the tree structure
        % T(j).Node = [T(j).Node, 2*Ssplit, 2*Ssplit+1]; % Store all the nodes in Tj
        T(j).Terminal = Tterminal; % store the index of terminal nodes
        T(j).Internal = Internal; % store the index of internal nodes
        T(j).spvar = [T(j).spvar,k]; % Store the split variable
        T(j).sprule =[T(j).sprule, splitvalue]; % Store the split rule
        fieldname ='t'+string(oldterminal(oldterminal==Ssplit));
        Terstr(j).(fieldname) = [];
        for tt = 1:length(newTterminal)            
            if rem(newTterminal(tt),2) == 0
                fieldname = 't'+string(newTterminal(tt));
                Terstr(j).(fieldname) = RLind;
            else
                fieldname ='t'+string(newTterminal(tt));
                Terstr(j).(fieldname) = RRind;
            end      
        end
    end
end