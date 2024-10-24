function [T,Terstr,Rj] = sample_change(T,Terstr,j,p,m,trainx,ytilta,mu,ntrain,  sigma2, sigmamu2)     
    % pick a parent node with 2 children
    Tterminal = T(j).Terminal;
    temp = Tterminal(rem(Tterminal,2)==0);
    parent = temp(ismember(Tterminal(rem(Tterminal,2)==0)+1,Tterminal))/2;
    changeprob = (1/length(parent):1/length(parent):1);
    parentindex = parent(find(changeprob>rand,1));
    
    % find the corresponding dataset
    leftind = 2*parentindex; rightind = 2*parentindex+1;
    childindex = [leftind,rightind];
    fieldname = 't'+string(childindex(1));
    subindex1 = Terstr(j).(fieldname);
    fieldname = 't'+string(childindex(2));
    subindex2 = Terstr(j).(fieldname);
    subindex = [subindex1; subindex2];
    
    xgrow = trainx(subindex,:); % this is the subdata set on the parent node
    
    % find the original RL and RR
    Rj = computeRj(T, Terstr,m,mu,j,ytilta,ntrain);
    Rl = Rj(subindex);
    RL = Rj(subindex1); nL = length(RL);
    RR = Rj(subindex2); nR = length(RR);
     
    % find the covariantes that contain more than 2 unique values
%     varnum = 0;
%     varlist = zeros(p,1);
%      for pind = 1:p
%          [~,ia,~] = unique(xgrow(:,pind));
%          nunique = length(ia);
%          if nunique >= 2
%              varnum = varnum+1;
%              varlist(pind) = pind;
%          end
%      end
%      varlist = nonzeros(varlist);
     varlist = find( arrayfun( @(c) numel( unique( xgrow(:, c) )), 1:p) >= 2 )';
     varnum = length(varlist);
     
     % pick a new splitting variable and a splitting rule
     probx = (1/varnum:1/varnum:1);
     svstar = varlist(find(probx>rand,1));
     [~,ia,~] = unique(xgrow(:,svstar));
     neta = length(ia);
     probvalue = (1/neta:1/neta:1);
     e=find(probvalue>rand,1);
     srstar = xgrow(ia(e),svstar);
     
     % compute the corresponding new RL, RR  
     tmp = sort(unique(xgrow(:,svstar)));
     if tmp==2
         RLind = subindex(xgrow(:,svstar) == tmp(1));
         RLstar = Rl(xgrow(:,svstar) == tmp(1)); 
         nLstar = length(RLstar);
         RRind = subindex(xgrow(:,svstar) == tmp(2));
         RRstar = Rl(xgrow(:,svstar) == tmp(2));
         nRstar = length(RRstar);
     else
         RLind = subindex(xgrow(:,svstar)<= srstar);
         RRind = subindex(xgrow(:,svstar)> srstar);
         RLstar = Rl(xgrow(:,svstar)<= srstar); nLstar = length(RLstar);
         RRstar = Rl(xgrow(:,svstar)> srstar); nRstar = length(RRstar);
     end
     
     % compute the M-H ratio
     logBc = 1/2*(log(sigma2/sigmamu2+nL)+log(sigma2/sigmamu2+nR)-log(sigma2/sigmamu2+nLstar)-log(sigma2/sigmamu2+nRstar))+...
         (1/(2*sigma2))*((sum(RLstar))^2/(nLstar+sigma2/sigmamu2)+(sum(RRstar))^2/(nRstar+sigma2/sigmamu2)-(sum(RL))^2/(nL+sigma2/sigmamu2)-(sum(RR))^2/(nR+sigma2/sigmamu2));
     
     % accept or not?
     uc = rand;
     if log(uc)<=logBc % accept this change. change the corresponding splitting rule and splitting variables. change the corresponding index in Terstr
         tmp = T(j).Internal;
         T(j).spvar(tmp == parentindex) = svstar;
         T(j).sprule(tmp == parentindex)= srstar;

         for tt = 1:2
             if rem(childindex(tt),2) == 0
                 fieldname = 't'+string(childindex(tt));
                 Terstr(j).(fieldname) = RLind;
             else
                 fieldname = 't'+string(childindex(tt));
                 Terstr(j).(fieldname) = RRind;
             end
         end
     end
     
end