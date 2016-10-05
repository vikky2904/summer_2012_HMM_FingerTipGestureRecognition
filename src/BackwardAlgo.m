function [Betaa] = BackwardAlgo(Pi, a, b, Ob, O)

% Backward Algorithm

%% Setting up matrices and variables

% a: NxN matrix = Prob. of transition from state Si to state Sj
% b: NxM matrix = Prob. of observing Vk for state Si
% Ob : Mx1 matrix : Vector of all possible observations
% O : EGxT matrix : Matrix of EG no. of 1xT-dimensional observation sets

N = size(a,1);
M = size(Ob,1);
T = size(O,2);
EG = size(O,1);

Betaa = zeros(N,T); BetaNew = Betaa;

%% Adding up the normalised Beta values for each example, found using the Backward Algo

for eg = 1:EG %for loop to run through all the examples
    
%% Initialization

% Beta: NxT matrix
% Beta(i,find(Ob==O(eg,t),1)) = Probability of partial obs. seq. from t+1 to end, given state Si at t
% Beta(i,T) = 1

%{
for i=1:N    
    Betaa(i,T) = 1;    
end
%}

BetaNew(:,T) = 1;
Betaa(:,T) = Betaa(:,T) + BetaNew(:,T);

%% Induction

% Beta(i,t) = ( sum w.r.t. j a(i,j)*b(j,find(O(t+1)))*Beta(j,t+1) )

for t = (T-1):-1:1
    BetaNew(:,t) = a*( b(:,Ob==O(eg,t+1)).*BetaNew(:,t+1) );
    %Normalising BetaNew(:,t) and adding it to Beta(:,t)
    c(t) = sum(BetaNew(:,t));
	Betaa(:,t) = Betaa(:,t) + BetaNew(:,t)/c(t);
end


end %end of for loop running through all the examples

%% Finding average of Beta by diving by number of examples

Betaa = Betaa/EG;

end

