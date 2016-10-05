function [] = FingerTipGestureRecognition()
%% MATLAB Code for Finger-Tip Gesture Recognition using two Hidden Markov Models

% Recognises the gestures Triangle, Square, and Diamond,
% performed by the tip of a finger.
%   -> Please ensure the background does not have skin-coloured portions.
%   -> First, a preview of the video to be recorded is shown for about 2
%   seconds, along with a new blank window kept open.
%   -> Position your finger only in the left half of the preview video.
%   -> Once the words 'GO' appear on the blank window, strat your gesture.
%   -> The window will now show a mirror image of the preview video, along with
%   tracking of the fingertip.
%       Make note of the limits of the window, and the half-line shown for convenience.
%       Only perform the gesture in the half your finger is in.
%   -> Perform the gesture for about 5 seconds.
%   -> The recognised gesture name shall be displayed on the screen.
%   -> The command window can also be checked to know the recognised gesture.

clear; close all;

imaqhwinfo

tts('hello');

% Matrix to store coordinates of finger-tip (will be used subsequently)
fingercoord = zeros(0,2);

% Defining a kernel for blurring images (will be used subsequently)
kernel = ones(5,5)/25;

% Setting up video
vid = videoinput('winvideo',1);
set(vid, 'ReturnedColorspace', 'RGB');

% Finding the size of the window frame
vidsize = get(vid, 'VideoResolution');
width = vidsize(1);
height = vidsize(2);

% Finding size of screen
screensize = get(0,'ScreenSize');

% Setting up a preview of the video to be recorded
figure('name', 'preview', 'Position', [1 screensize(4)/2 vidsize(1) vidsize(2)]);
im = image(zeros(height, width, 3));
line([.5*width .5*width], [1 height], 'color', 'g', 'LineWidth', 2);
preview(vid, im);

% Setting up a the video to be recorded as gesture
figure('name', 'RGBimage');
pause(2.5);
text(.3,.5,'GO', 'BackgroundColor' , 'white', 'Color', 'black', 'FontName', 'Impact', 'FontSize', 100);
pause(.001);
tts('go');
pause(.3);

count = 0;
tic;

while(count<27)
    
% Getting image as double
RGBimage = im2double(getsnapshot(vid));
% Taking mirror-image
RGBimage = RGBimage(:,size(RGBimage,2):-1:1,:);

%Convention of directions x and y in image is:
%Inc_x:DOWN, Inc_y:RIGHT

% Blurring the image
blur = RGBimage;
blurred = conv2(RGBimage(:,:,1), kernel);
blur(:,:,1) = blurred(3:(size(blurred,1)-2),3:(size(blurred,2)-2));
blurred = conv2(RGBimage(:,:,2), kernel);
blur(:,:,2) = blurred(3:(size(blurred,1)-2),3:(size(blurred,2)-2));
blurred = conv2(RGBimage(:,:,3), kernel);
blur(:,:,3) = blurred(3:(size(blurred,1)-2),3:(size(blurred,2)-2));
%figure('name', 'smoothed'), imshow(blur);

% Determining h s i values for each pixel
HSIimage = rgb2hsi(blur);

% Determining mask for skin-coloured regions:
% 0 <= Hue(0:360) <= 60, 10 <= Saturation(0:100) <= 40
mask = zeros(size(RGBimage(:,:,1)));
mask((HSIimage(:,:,1)>=0)&(HSIimage(:,:,1)<=60)&...
    (HSIimage(:,:,2)>=10)&(HSIimage(:,:,2)<=40)) = 1;
blurred = conv2(mask, kernel);
mask = blurred(3:(size(blurred,1)-2),3:(size(blurred,2)-2));
mask(mask<.5) = 0;
mask(mask>=.5) = 1;
%figure('name', 'mask'), imshow(mask);
%{
% Getting RGB image with only skin
skin = zeros(size(RGBimage));
skin(:,:,1) = RGBimage(:,:,1).*mask;
skin(:,:,2) = RGBimage(:,:,2).*mask;
skin(:,:,3) = RGBimage(:,:,3).*mask;
figure('name', 'skin'), imshow(skin);

% Grayscale skin image
grayskin = rgb2gray(skin);
figure('name', 'gray skin'), imshow(grayskin);
%}
% Determining mask for region of gesturing as the right half of the image
mask(:,1:.5*size(mask,2)) = 0;

% Finding the tip of the finger within the mask created
% as the first white pixel found from top
% in convention Inc_x:DOWN, Inc_y:RIGHT
[y x] = find(mask'>0,1);

% If the finger-tip is detected, and it lies within 3 pixels of the
% boundary of the right half of the image
if ~isempty(x) && y>.5*width+3 && y<width-3 && x>3 && x<height-3
        
    % Appending matrix 'fingercoord' with new coordinates of first white pixel
    fingercoord = [fingercoord; x y];
    
    % Showing image with path of first white pixel
    fingerpath = blur;
    for i = 1:size(fingercoord,1)
        fingerpath((fingercoord(i,1)-1):(fingercoord(i,1)+1),(fingercoord(i,2)-1):(fingercoord(i,2)+1),1) = 255/255;
        fingerpath((fingercoord(i,1)-1):(fingercoord(i,1)+1),(fingercoord(i,2)-1):(fingercoord(i,2)+1),2) = 63/255;
        fingerpath((fingercoord(i,1)-1):(fingercoord(i,1)+1),(fingercoord(i,2)-1):(fingercoord(i,2)+1),3) = 52/255;
    end
    
    % Showing region of white pixel found above as a 6x6 red region
    fingerpath((x-2):(x+2),(y-2):(y+2),1) = 1;
    fingerpath((x-2):(x+2),(y-2):(y+2),2) = 1;
    
    imshow(fingerpath);
    
    % Showing boundary of right half of image
    % Convention of directions x and y in image processing:
    % Inc_x:RIGHT, Inc_y:UP
    line([.5*width .5*width], [1 height], 'color', 'g', 'LineWidth', 2);
    line([width width], [1 height], 'color', 'g', 'LineWidth', 2);
    line([.5*width width], [1 1], 'color', 'g', 'LineWidth', 2);
    line([.5*width width], [height height], 'color', 'g', 'LineWidth', 2);
    
    count = count + 1;
end

%figure('name', 'mask'), imshow(mask);

toc;
pause(.100);

end

close all;
closepreview(vid);

% Displaying the whole path of the first white pixel found
fingerpath = blur;
for i = 1:size(fingercoord,1)
    fingerpath(fingercoord(i,1):(fingercoord(i,1)+1),fingercoord(i,2):(fingercoord(i,2)+1),1) = 1;
    fingerpath(fingercoord(i,1):(fingercoord(i,1)+1),fingercoord(i,2):(fingercoord(i,2)+1),2) = 0;
    fingerpath(fingercoord(i,1):(fingercoord(i,1)+1),fingercoord(i,2):(fingercoord(i,2)+1),3) = 0;
end
% Also, displaying the last fingertip
if ~isempty(x) && y>.5*width+3 && y<width-3 && x>3 && x<height-3
    fingerpath((x-2):(x+2),(y-2):(y+2),1) = 1;
    fingerpath((x-2):(x+2),(y-2):(y+2),2) = 1;
    fingerpath((x-3):(x+3),(y-3):(y+3),3) = 0;
end;
figure('name', 'fingerpath'), imshow(fingerpath);

%% HMM 1: To determine the state among Right, Down, Left, Up, DR, DL, UL, UR

% Possible States: Right, Down, Left, Up, DR, DL, UL, UR

% Matrix of the observations O
% Changing from convention of image: Inc_x:DOWN, Inc_y:RIGHT
% to convention of graphs: Inc_x:RIGHT, Inc_y:UP
O = [fingercoord(:,2) height-fingercoord(:,1)];

% Finding difference of current coordinates from previous ones to get an observation sequence
% that describes constancy or increase or decrease in x and y coordinates
Obs = sign(O - [O(1,:); O(1:(size(O,1)-1),:)]);

% Deleting the first 4 and the last 2 observations, 
% assuming them to be not part of the gesture
Obs = reshape(Obs(5:(size(Obs,1)-2),:),size(Obs,1)-6,size(Obs,2));

% Possible Observations: [NoChangeIn_x NoChangeIn_y; Dec_x Dec_y; Inc_x Inc_y]
Ob = [ 0 0; -1 -1; 1 1];

% Initial Probability matrix
Pi = [.4; .03; .03; .04; .4; .03; .03; .04];

% Transition matrix
a = [.6 .25 .01 .01 0.1 .01 .01 .01; .01 0.6 .25 .01 0.1 .01 .01 .01; .01 .01 .65 .15 .01 .01 .01 .15; .01 .01 .01 .93 .01 .01 .01 .01;...
    .01 .01 .15 .01 .65 .15 .01 .01; .01 .05 .05 .01 .01 0.6 .25 .01; .01 .01 .055 .055 .01 .01 0.6 .25; .01 .01 .01 .01 .01 .01 .01 .93];

% Emission matrix
b(:,:,1) = [.1 0 .9; .34 .33 .33; .1 .9 0; .34 .33 .33; .1 0 .9; .1 .9 0; .1 .9 0; .1 0 .9]; % for x coordinates
b(:,:,2) = [.34 .33 .33; .1 .9 0; .34 .33 .33; .1 0 .9; .1 .9 0; .1 .9 0; .1 0 .9; .1 0 .9]; % for y coordinates

% Finding the sequence of states for the different observations
% using the Viterbi algorithm
% V = ViterbiAlgo(InitialProbMatrix, TransitionMatrix, EmissionMatrix,...
%   PossibleOutputs, Observations)
V = ViterbiAlgo(Pi, a, b, Ob, Obs);

%% HMM 2: To determine the state among Triangle, Square, Diamond

% Possible States: Triangle, Square, Diamond

% Observation sequence: Sequence of states found in previous HMM

% Possible Observations: [R; D; L; U; DR; DL; UL; UR]
OB = [1; 2; 3; 4; 5; 6; 7; 8];

% Initial Probability matrix
PI = [.34; .33; .33];

% Transition matrix
A = [.98 .01 .01; .01 .98 .01; .01 .01 .98];

% Emission matrix
B = [.09 .04 .25 .04 .25 .04 .04 .25; 0.22 0.22 0.22 0.22 .03 .03 .03 .03; .03 .03 .03 .03 0.22 0.22 0.22 0.22];

% Determiining the gesture by finding the state in which the system was,
% using the Viterbi algorithm
%Q = ViterbiAlgo(prior, transmat, obsmat, OB, V);
Q = ViterbiAlgo(PI, A, B, OB, V);

if size(unique(Q))==1
    switch unique(Q)
        case 1
            disp('Triangle!');
            text(1,.7*width,'Triangle!', 'BackgroundColor' , 'white', 'Color', 'black', 'FontName', 'Arial', 'FontSize', 15);
            tts('Triangle!');
        case 2
            disp('Square!');
            text(1,.7*width,'Square!', 'BackgroundColor' , 'white', 'Color', 'black', 'FontName', 'Courier New', 'FontSize', 15);
            tts('Square');
        case 3
            disp('Diamond!');
            text(1,.7*width,'Diamond!', 'BackgroundColor' , 'white', 'Color', 'black', 'FontName', 'Harrington', 'FontSize', 15);
            tts('Diamond!');
    end
else
    disp('Are you kidding me..?');
    text(1,.7*width,'Are you kidding me..?', 'BackgroundColor' , 'white', 'Color', 'black', 'FontName', 'Impact', 'FontSize', 15);
    tts('Are you kidding me?');
end

end

%% RGB to HSI
function [HSIimage] = rgb2hsi(RGBimage)
% rgb2hsi: Converts an image in rgb
% color space to hsi color space
%   Converts Red,Green,Blue values to 
%   Hue (in degrees, from 0 to 360), 
%   Saturation (as percentage, from 0 to 100),
%   Intensity values (in gray level unit, from 0 to 255)

pi = 3.1416;
RGBimage = im2double(RGBimage);
R = RGBimage(:,:,1);
G = RGBimage(:,:,2);
B = RGBimage(:,:,3);

I = R + B + G;
i = I/3;

r = R./I; g = G./I; b = B./I;

w = .5*(R-G + R-B)./sqrt((R-G).^2 + (R-B).*(G-B));
w(w>1) = 1;
w(w<-1) = -1;

h = acos(w); s = h;

h(B>G) = 2*pi - h(B>G);
h((R==G)&(G==B)) = 0;
h = h*180/pi;
    
s((r<=g)&(r<=b)) = 1 - 3*r((r<=g)&(r<=b));
s((g<=r)&(g<=b)) = 1 - 3*g((g<=r)&(g<=b));
s((b<=r)&(b<=g)) = 1 - 3*b((b<=r)&(b<=g));
s((R==G)&(G==B)) = 0;
s = s*100;

HSIimage = RGBimage;
HSIimage(:,:,1) = h;
HSIimage(:,:,2) = s;
HSIimage(:,:,3) = i;


end

%% Viterbi Algorithm
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

end

%% Text-to-Speech
function tts( text )
% This function converts text into speech.
% You can enter any form of text (less than 512 characters per line) into
% this function and it speaks it all.
%
% Note: Requires .NET
%
% Input:
% * text ... text to be spoken (character array, or cell array of characters)
%
% Output:
% * spoken text
%
% Example:
% Casual chat.
% Speak({'Hello. How are you?','It is nice to speak to you.','regards SAPI.'})
%
% Emphasising, silence, pitching, ... can be done (see external links)
%
% TODO: allow the above mentioned changes in voice
%
% See also: initSpeech, unloadSpeechLibrary
%
% External
% Microsoft's TTS Namespace
% http://msdn.microsoft.com/en-us/library/system.speech.synthesis.ttsengine(v=vs.85).aspx
% Microsoft's Synthesizer Class
% http://msdn.microsoft.com/en-us/library/system.speech.synthesis.speechsynthesizer(v=vs.85).aspx
%
%% Signature
% Author: W.Garn
% E-Mail: wgarn@yahoo.com
% Date: 2011/01/25 12:20:00 
% 
% Copyright 2011 W.Garn
%
if nargin<1
    text = 'Please call this function with text';
end
try
    NET.addAssembly('System.Speech');
    Speaker = System.Speech.Synthesis.SpeechSynthesizer;
    if ~isa(text,'cell')
        text = {text};
    end
    for k=1:length(text)
        Speaker.Speak (text{k});
    end
catch
    warning(['If this is not a Windows system or ' ...
        'the .Net class exists you will not be able to use this function.' ...
        'Please let me know what went wrong: wgarn@yahoo.com']);
end

end
