function [pathFamily,score,D,SSM] = computePathFamily(segment,SSM_full,Parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: computePathFamily.m
% Date of Revision: 2013-06
% Programmer: Nanzhu Jiang, Peter Grosche, Meinard Müller
% http://www.audiolabs-erlangen.de/resources/MIR/SMtoolbox/
%
%
% Description:
%   Computes an optimal path family over segment from a self-similarity
%   matrix SSM.
%
% Input: segment alpha=[s,t] given by start s and end t index
%        Self similarity matrix SSM
%
% Output: optimal pathfamily
%         score : score of pathfamily
%       
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Reference:
%   If you use the 'SM toobox' please refer to:
%   [MJG13] Meinard Müller, Nanzhu Jiang, Harald Grohganz
%   SM Toolbox: MATLAB Implementations for Computing and Enhancing Similarity Matrices
%   Proceedings of the 53rd Audio Engineering Society Conference on Semantic Audio, London, 2014.
%
% License:
%     This file is part of 'SM Toolbox'.
%
%     'SM Toolbox' is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 2 of the License, or
%     (at your option) any later version.
%
%     'SM Toolbox' is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with 'SM Toolbox'. If not, see
%     <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



if nargin < 2
    error('You must at least provide a segment and a self similarity matrix.');
end

if nargin < 3
    Parameter = [];
end

if ~isnumeric(SSM_full) || ~isreal(SSM_full)
    error('The self similarity matrix must be numeric but not complex');
end

% steps in n direction
if isfield(Parameter,'dn')
    dn = Parameter.dn;
else
    dn = int32([1 0 1]);
end

% steps in m direction
if isfield(Parameter,'dm')
    dm = Parameter.dm;
else
    dm = int32([0 1 1]);
end

% weights for the steps, untested
if isfield(Parameter,'dw')
    dw = Parameter.dw;
else
    dw = [1 1 1];
end

if any(segment < 1) || segment(1)>segment(2)
    error('Segment must contain only integers greater than one and segment(2) must be larger or equal than segment(2)');
end

if ~isinteger(dn) || any(dn < 0) || all(dn == 0)
    error('The dn parameter must contain only integers greater or equal zero and at least one integer greater zero');
end

if ~isinteger(dm) || any(dm < 0) || all(dm == 0)
    error('The dm parameter must contain only integers greater or equal zero and at least one integer greater zero');
end

if ~isnumeric(dw) || any(dw <= 0)
    error('The dw parameter must contain only numbers greater than zero');
end



N = int32(size(SSM_full,1))+1;
M = int32(segment(2)-segment(1)+1)+1;
S = int32(size(dn,2));
segment = double(segment);

if S ~= size(dm,2) || S ~=  size(dw,2)
    error('The parameters dn,dm, and dw must be of equal length.');
end

%% calc bounding box size of steps, for inf-padding
sbbn = max(dn);
sbbm = max(dm);



% one special column and row
SSM = zeros(N,M);
SSM(1:N-1,2:M) = SSM_full(:,segment(1):segment(2));
E = zeros(N,M,'int8');




%% initialize extended D matrix, inf-padded

D = -inf(sbbn+N,sbbm+M);
% D(1+sbbn,1+sbbm) = SSM(1,2);
D(sbbn,1+sbbm) = 0;


%% accumulate
for n=(1:N)+sbbn
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % m=1 is special column with special steps:
    %  
    %    (-1,0): going up with zero score
    %    (-1,M): jumping back from t to s
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    m = 1+sbbm;
    [v,i] = max([D(n-1,m),D(n-1,M+sbbm)]);
    D(n,m) = v;  %Note: SSM(n,1)= 0
    if i==1 % up-lift
        E(n-sbbn,m-sbbm) = 0;
    else % jump from t to s
        E(n-sbbn,m-sbbm) = -1;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % m=2 is another special case as we here allow only horizontal steps
    % from the special column m=1
    %  
    %    (0,-1): coming from left
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    m = 2+sbbm;
    D(n,m) = D(n,m-1) + SSM(n-sbbn,m-sbbm);
    E(n-sbbn,m-sbbm) = -2;
    score = -1;
    step_n = -1;
    step_m = -1;
    s = -1;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % m>2 is ordinary DTW with given stepsizes
    %  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for m=(2:M)+sbbm
        for s=1:S
            step_n = n-dn(s);
            step_m = m-dm(s);
            
            if step_m>1+sbbm

                score = D(step_n,step_m)+SSM(n-sbbn,m-sbbm)*dw(s);
    %             [D(n,m),Idx] = max([D(n,m) score]);

                if score>D(n,m)
                    D(n,m) = score;
                    E(n-sbbn,m-sbbm) = s;
                end
            end
        end
    end
end

% revert inf-padding
D = D((1:N)+sbbn,(1:M)+sbbm);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Backtracking of score maximizing path
% until n = m = 1
% this code ois related to TH_DTW_E_to_Warpingpath.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% end position is (N,1)
n = N;

m = 1;

score = D(n,m);

numPath = 0;
pathFamily = cell(0,0);


len = 0;
while true
    if E(n,m) == 0
        n = n-1;
    elseif E(n,m) == -1 % we jumped -> a path ends
        n = n-1;
        m = M;
        if numPath>0 %Store and clean up path
            p = p(:,1:len);
            p = fliplr(p);
            p(2,:)=p(2,:)+segment(1)-1;
            pathFamily{numPath,1} = p;
        end
        numPath = numPath+1;
        p = zeros(2,N+M+1);
        len = 0;
    elseif E(n,m) == -2 % we left the special column
        m = m-1;
        n = n;
    else
        stepidx = E(n,m);
        m = m-dm(stepidx);
        n = n-dn(stepidx);
    end
    if m>1
        len = len + 1;
        p(:,len) = [n, m-1]; %compensate for first zero column
    end
    if n <= 1 && m <= 1
        break;
    end
end

p = p(:,1:len);
p = fliplr(p);
p(2,:)=p(2,:)+segment(1)-1;
pathFamily{numPath,1} = p;


endfunction



