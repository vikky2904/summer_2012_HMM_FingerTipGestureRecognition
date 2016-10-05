function [PiNew, aNew, bNew, AlphaNew, BetaNew, logProb, lP] =  BaumWelsh(Pi, a, b, Ob, O, Alpha, Betaa, iters, maxIters, oldLogProb, lP)

%% Reestimation of parameters using Baum-Welch method, or the EM method

% Pi : Nx1 matrix : Vector of initial probabilities of states
% a : NxN matrix : Prob. of transition from state Si to state Sj
% b : NxM matrix : Prob. of observing Vk for state Si
% Ob : Mx1 matrix : Vector of all possible observations
% O : EGxT matrix : Matrix of EG no. of 1xT dimensional observation sets
% Alpha : NxT matrix; Alpha(i,t) = Probability of the partial Obs. seq. O1,O2,...,Ot, and state Si at t
% Betaa : NxM mmatrix; Betaa(i,find(Ob==O(eg,t),1)) = Prob. of partial obs. seq. Ot+1,...,OT, given state Si at t, in example eg

N = size(a,1);
M = size(Ob,1);
T = size(O,2);
EG = size(O,1);

% 1)Loop though all the examples, add up all the normalised Pi, a and b found, 
% 2)and finally take their average (divide them by number of examples)

PiNew = zeros(N,1);
aNew = zeros(N,N); aN = zeros(N,N);
bNew = zeros(N,M); bN = zeros(N,N);

%% 1)Loop though all the examples, add up all the normalised Pi, a and b found
for eg = 1:EG  

%% Xi: NxNxT matrix
% Xi(i,j,t) = Probability of being in state Si at t and Sj at t+1
% denominator = sum thru i: sum thru j : Alpha(i,t) * a(i,j) * b(j,find(O(eg,t+1))) * Betaa(j,t+1)
% Xi(i,j,t) = ( Alpha(i,t) * a(i,j) * b(j,find(O(eg,t+1))) * Betaa(j,t+1) )/denominator
Xi = zeros(N,N,T);

for t = 1:(T-1)
    den = sum( Alpha(:,t) .* ( a * ( b(:,Ob==O(eg,t+1)).*Betaa(:,t+1)) ) );
    Xi(:,:,t) = a.*((b(:,Ob==O(eg,t+1)).*Betaa(:,t+1))*Alpha(:,t)')'/den;
end


%% Gamma: NxT matrix
% Gamma(i,t) = Probability of being in state Si at t
% Gamma(i,t) = sum thru j: xi(i,j,t)
Gamma = reshape(sum(Xi,2), [size(Xi,1) size(Xi,3)]);
Gamma(:,T) = Alpha(:,T).*Betaa(:,T)/sum(Alpha(:,T).*Betaa(:,T));


%% Parameters

% Pi: Nx1 matrix
% Pi(i) = Expected number of times in state Si at t = gamma(i,1)
sumGamma = sum(Gamma(:,1)); %for normalisation
%Normalising Gamma(:,1) and adding it to PiNew:
PiNew = PiNew + Gamma(:,1)/sumGamma;

% a: NxN matrix
% a(i,j) = Prob. of transition from state Si to state Sj
% a(i,j) = (Expected no. of transitions from Si to Sj)/(Expected no. of transitions from Si)
%        = (sum w.r.t. t thru 1:(T-1) of:  Xi(i,j,t))/(sum w.r.t. t thru 1:(T-1) of: Gamma(i,t))
numr = sum(Xi(:,:,1:(T-1)),3);
sumG = (sum(Gamma(:,1:(T-1)),2)); sumG = repmat(sumG, 1, N);
aN = numr./sumG;
%Normalising aN and adding it to aNew:
aNewSum = sum(aN,2); aNewSum = repmat(aNewSum, 1, N);
aNew = aNew + aN./aNewSum;

% b: NxM matrix
% b(i,m) = Prob. of observing Vm for state Si
% b(i,m) = (Expected no. of times in state Si and obs. Vm)/(Expected no. of time in state Si)
%        = (sum w.r.t. t of: Gamma(i,t) s.t. Ot = Vm)/(sum w.r.t t of: Gamma(i,t))
for m = 1:M
	bN(:,m) = (Gamma*(O(eg,:)==Ob(m))')./sum(Gamma,2);
end
%Normalising bN and adding it to bNew:
bNewSum = sum(bN,2); bNewSum = repmat(bNewSum, 1, M);
bNew = bNew + bN./bNewSum;


%%
end % end of for loop going throught the examples

PiNew = PiNew/EG;
aNew = aNew/EG;
bNew = bNew/EG;

%% Finding new Alpha and Betaa

[AlphaNew, c] = ForwardAlgo(PiNew, aNew, bNew, Ob, O);
BetaNew = BackwardAlgo(PiNew, aNew, bNew, Ob, O);


%% LogProb

iters = iters + 1;

% Log likelihood = sum thru t (log(sum(alpha(:,t)))
logProb = sum(log(c));
lP = [lP; logProb];

fprintf('iter# %d  oldLogProb %f  logProb %f\n', iters, oldLogProb, logProb);

if iters < maxIters
    diff = abs(logProb - oldLogProb);
    avg = (abs(logProb) + abs(oldLogProb) + eps)/2;
    if diff/avg < 1e-4
        disp('converged');
        return;
    else
        oldLogProb = logProb;
        [PiNew aNew bNew AlphaNew BetaNew logProb lP] = BaumWelsh(PiNew, aNew, bNew, Ob, O, AlphaNew, BetaNew, iters, maxIters, oldLogProb, lP);
    end
end

%{
if iters < maxIters
    converged = em_converged(logProb, oldLogProb);
    if ~converged
        fprintf('iters: %d, ~c\n', iters);
        oldLogProb = logProb;
        [PiNew aNew bNew AlphaNew BetaNew logProb lP] = BaumWelsh(PiNew, aNew, bNew, Ob, O, AlphaNew, BetaNew, iters, maxIters, oldLogProb, lP);
    end
end
%}

end

%% EXAMPLE
%{

N = 3;

load('Obs.mat');

O = -ones(size(Obs_x));
for i = 1:length(Obs_x)
    O(i, Obs_x(i,:)==  0 & Obs_y(i,:)==  0) = 1;
    O(i, Obs_x(i,:)==  0 & Obs_y(i,:)== -1) = 2;
    O(i, Obs_x(i,:)==  0 & Obs_y(i,:)==  1) = 3;
    O(i, Obs_x(i,:)== -1 & Obs_y(i,:)==  0) = 4;
    O(i, Obs_x(i,:)== -1 & Obs_y(i,:)== -1) = 5;
    O(i, Obs_x(i,:)== -1 & Obs_y(i,:)==  1) = 6;
    O(i, Obs_x(i,:)==  1 & Obs_y(i,:)==  0) = 7;
    O(i, Obs_x(i,:)==  1 & Obs_y(i,:)== -1) = 8;
    O(i, Obs_x(i,:)==  1 & Obs_y(i,:)==  1) = 9;
end

Ob = [1;2;3;4;5;6;7;8;9];
M = size(Ob,1);

Pi = rand(N,1);
Pi = Pi/sum(Pi);

a = rand(N,N);
a = a./repmat(sum(a,2), [1 N]);

b = rand(N,M);
b = b./repmat(sum(b,2), [1 M]);

iters = 0;
maxIters = 100;
oldLogProb = -Inf;
lP = zeros(0,1);

Alpha = ForwardAlgo(Pi, a, b, Ob, O);
Betaa = BackwardAlgo(Pi, a, b, Ob, O);

[PiNew, aNew, bNew, AlphaNew, BetaNew, logProb, lP] = BaumWelsh(Pi, a, b, Ob, O, Alpha, Betaa, iters, maxIters, oldLogProb, lP);

%}
	