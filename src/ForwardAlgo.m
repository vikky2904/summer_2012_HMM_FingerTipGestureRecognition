function [Alpha, c, P] = ForwardAlgo(Pi, a, b, Ob, O)

%% Forward Algorithm

%% Setting up matrices and variables

% Pi: Nx1 matrix : Vector of initial probabilities of states
% a: NxN matrix : Prob. of transition from state Si to state Sj
% b: NxM matrix : Prob. of observing Vk for state Si
% Ob : Mx1 matrix : Vector of all possible observations
% O : EGxT matrix : Matrix of EG no. of 1xT dimensional observation sets

N = size(a,1);
M = size(Ob,1);
T = size(O,2);
EG = size(O,1);

c = zeros(T,1); cNew = c;
Alpha = zeros(N,T); AlphaNew = Alpha;


%% Adding up the normalised Alpha values for each example found using the Forward Algo

for eg = 1:EG %for loop to run through all the examples

%% Initialization

% Alpha: NxT matrix
% Alpha(i,t) = Probability of the partial Obs. seq. O1,O2,...,Ot, and state Si at t
% Alpha(i,1) = Pi(i) * b(i,find(Ob == O(eg,1),1))

AlphaNew(:,1) = Pi .* b(:, Ob==O(eg,1));

%Normalising AlphaNew and adding to the Alpha matrix:
cNew(1) = sum(AlphaNew(:,1));
Alpha(:,1) = Alpha(:,1) + AlphaNew(:,1)/cNew(1);


%% Induction

% Alpha(j,t) = ( sum w.r.t. i (Alpha(i,t-1)*a(i,j)) ) * b(j, find(Ob==O(eg,t),1))

for t = 2:T
    for j = 1:N
        AlphaNew(j,t) = sum(AlphaNew(:,t-1).*a(:,j)) * b(j, Ob==O(eg,t));
    end
    cNew(t) = sum(AlphaNew(:,t));

    %Normalising AlphaNew(:,t) and adding it to Alpha(:,t)
	Alpha(:,t) = Alpha(:,t) + AlphaNew(:,t)/cNew(t);
end

c = c + cNew;


%% Termination

% P(O|parameters) = sum w.r.t. i of: Alpha(i,T)

%P = sum(Alpha(:,T));
P = Alpha(:,T);

end %end of for loop running through all the examples 
 
%% Taking avg of all the Alpha's by diving by number of examples

Alpha = Alpha/EG;

end

