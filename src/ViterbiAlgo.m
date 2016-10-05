function [Q] = ViterbiAlgo(Pi, a, b, Ob, O)

% Viterbi Algorithm 
%   To determine the most probable state sequence Q,
%   given Pi: the Initial Probability matrix, a: Transition matrix,
%   b: Emission matrix, Ob: matrix of Possible Observations, and
%   O: matrix of observations (one observation set)

%% Setting up matrices and variables

% Pi: Nx1 matrix : Vector of initial probabilities of states
% a: NxN matrix : Prob. of transition from state Si to state Sj
% b: NxM matrix : Prob. of observing Vk for state Si
% Ob : Mx1 matrix : Vector of all possible observations
% O : Txr matrix : Matrix of T r-dimensional observations (only 1 observation set)

N = size(a,1);
M = size(Ob,1);
T = size(O,1);
r = size(O,2);

% delta: NxT matrix : delta(i,t) = Highest prob. along a path for first t obs. and ending in Si
Delta = zeros(N,T);

% psi: NxT matrix : argmax(delta)
Psi = Delta;

% Q: Tx1 matrix: vector of the most probable state sequence
Q = zeros(T,1);

bprod = ones(N,T);
for t = 1:T
    for ri = 1:r
        bprod(:,t) = bprod(:,t) .* b(:,Ob(:,ri)==O(t,ri),ri);
    end
end


%% Initialization

%Delta(:,1) = Pi .* b(:, Ob==O(1));
%Delta(:,1) = Pi .* b_1(:, find(Ob==O(1,1))).*b_2(:, find(Ob==O(1,2)));
Delta(:,1) = Pi .* bprod(:,1);
Psi(:,1) = 0;


%% Recursion

% delta(j,t) = max w.r.t. i  delta(i,t-1) * a(i,j) * b(j,find(O(t)))
% psi(j,t) = argmax w.r.t. i  delta(i,t-1) * a(i,j) * b(j,find(O(t)))

for t=2:T
    for j=1:N
        [Delta(j,t) Psi(j,t)] = max( Delta(:,t-1) .* a(:,j) * bprod(j,t) );
    end
end

%max( Delta(:,t-1) .* a(:,j) * b(j,find(Ob==O(t),1)) )
%max( Delta(:,t-1) .* a(:,j) * bprod(j,t) )

%% Termination

[~, Q(T)] = max(Delta(:,T));


%% Path back-tracking

for t=(T-1):-1:1
    Q(t) = Psi(Q(t+1),t+1);
end

