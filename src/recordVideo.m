clear all;

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

% Matrix to store coordinates of finger-tip (will be used subsequently)
fingercoord = zeros(50,2);

% Defining a kernel for blurring images (will be used subsequently)
kernel = ones(5,5)/25;

% Setting up a the video to be recorded as gesture
figure('name', 'RGBimage'), shg;
pause(2.5);
text(.3,.5,'GO', 'BackgroundColor' , 'white', 'Color', 'black', 'FontName', 'Impact', 'FontSize', 100);
pause(.3);

% Create a new AVI file
aviObject = avifile('1.avi');

count = 0;

while(count<120)
    
% Getting image as double
RGBimage = im2double(getsnapshot(vid));
% Taking mirror-image
RGBimage = RGBimage(:,width:-1:1,:);

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
mask = zeros(height, width);
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
[y x] = find(mask'>0,1);

%fingerpath = RGBimage;

% If the finger-tip is detected, and it lies within 3 pixels of the
% boundary of the right half of the image
fingerpath = blur;
if ~isempty(x) && y>.5*width+3 && y<width-3 && x>3 && x<height-3
    
    count = count + 1;
    
    % If within first two frames, take avg as the starting pt coords
    if count==2
        startcoords = (fingercoord(1,:) + [x y])/2;
    end
    
    % Appending matrix 'fingercoord' with new coordinates of first white pixel
    fingercoord(count,:) = [x y];
    
    % Showing image with path of first white pixel detected
    for i = 1:count
        fingerpath((fingercoord(i,1)-1):(fingercoord(i,1)+1),(fingercoord(i,2)-1):(fingercoord(i,2)+1),1) = 255/255;
        fingerpath((fingercoord(i,1)-1):(fingercoord(i,1)+1),(fingercoord(i,2)-1):(fingercoord(i,2)+1),2) = 63/255;
        fingerpath((fingercoord(i,1)-1):(fingercoord(i,1)+1),(fingercoord(i,2)-1):(fingercoord(i,2)+1),3) = 52/255;
    end
    
    % Showing region of white pixel found above as a 6x6 red region
    fingerpath((x-2):(x+2),(y-2):(y+2),1) = 1;
    fingerpath((x-2):(x+3),(y-2):(y+2),2) = 1;
    
    imshow(fingerpath);
    
    % Showing boundary of right half of image
    %Convention of directions x and y in image processing:
    %Inc_x:RIGHT, Inc_y:UP
    line([.5*width .5*width], [1 height], 'color', 'g', 'LineWidth', 2);
    line([width width], [1 height], 'color', 'g', 'LineWidth', 2);
    line([.5*width width], [1 1], 'color', 'g', 'LineWidth', 2);
    line([.5*width width], [height height], 'color', 'g', 'LineWidth', 2);
    
    if count>20 && sum(([x y]-startcoords).^2)<50
        break;
    end
    
end

fingerpath(fingerpath>1) = 1;

% Convert I to a movie frame
frame = im2frame(fingerpath);
% Add the frame to the AVI file
aviObject = addframe(aviObject, frame);

%figure('name', 'mask'), imshow(mask);

count
%pause(.070);

end

for i = 1:25
    aviObject = addframe(aviObject, frame);
end

aviObject = close(aviObject);

closepreview;
close all;
disp('done');

