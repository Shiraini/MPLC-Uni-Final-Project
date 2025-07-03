clear all;

%Set this to the factor you want to enlarge the input/output basis
%dimensions by e.g. to change the default input SMF MFD from 60um and HG
%output MFD of 400um to 120 and 800um, the rescaleFactor would be 2. When
%the scale of the basis is changed, the spacing between planes will also
%have to change (grows with the square. e.g. doubling the size of the
%basis, will quadruple the spacing between planes, 25mm-->100mm)
rescaleBasisFactor = 1;

%Pixel size at which the masks were originally calculated
%8um (Holoeye PLUTO)
pixelSizeNative = rescaleBasisFactor.*8e-6;

%New piSLM pixel size
pixelSizeNew = 9.2e-6;
%Pixel dimensions of new SLM
Nx = 1920;
Ny = 1200;
SLM = zeros(Nx,Ny,'single');

%Distance from 1 mask to the next on the SLM (in pixels)
maskSpacingPixels = 230;
%Distance from 1 mask to the next on the SLM (in metres)
maskSpacingMetres = maskSpacingPixels.*pixelSizeNew;
%This script tries to guess what the background level of scattered light is
%for the mask by looking at the average power of the pixels at top and
%bottom of the mask. You can then set any regions of the masks that have
%intensity less than the scatterThreshold*(estimated background scatter
%level) to zero. Then in turn you can set the background of the SLM to a
%pattern like a tilted grating such that any spurious scattered light that is
%due to finite spatial resolution of the SLM (e.g. phase wraps) ccan be
%removed from the system and tilted away from the rest of the beam.
scatterThreshold = 0;

%Load the actual masks
load('.\PhaseMasks.mat','MASKS');
s = size(MASKS);
%Total number of masks
maskCount = s(1);
%Pixel dimensions of masks
nx = s(2);
ny = s(3);

%Print out the corresponding mirror width that should be used given the
%size and number of masks
fprintf('Ideal mirror width %3.3f mm\n',(maskCount-1).*maskSpacingMetres.*1000);

%Setup the x-y co-ordinate system of the original masks
X0 = ((1:nx)-(nx/2+1)).*pixelSizeNative;
Y0 = ((1:ny)-(ny/2+1)).*pixelSizeNative;
[X0 Y0] = meshgrid(X0,Y0);

%Setup the x-y co-ordinate system of the new SLM
X1 = ((1:Nx)-(Nx/2+1)).*pixelSizeNew;
Y1 = ((1:Ny)-(Ny/2+1)).*pixelSizeNew;
[X1 Y1] = meshgrid(X1,Y1);

%For every mask...
for maskIdx=1:maskCount
    %Get the old mask
    MSK0 = squeeze(MASKS(maskIdx,:,:));
    %Calculate the centre position of this mask on the SLM
    dx = ((maskIdx-0.5)-maskCount/2.0).*maskSpacingMetres;
    %Estimate the background scatter level (few pixels at the edges)
    scatterLevel=(mean(mean(abs(MSK0(:,[1:4 (ny-3):ny])).^2)));
    %Set any regions below the threshold to zero
    MSK0(abs(MSK0).^2<scatterLevel.*scatterThreshold) = 0;
    %Interpolate the old mask, onto the new x-y coordinate system
    MSK1r = interp2(X0,Y0,real(MSK0).',X1-dx,Y1);
    MSK1i = interp2(X0,Y0,imag(MSK0).',X1-dx,Y1);
    MSK1 = MSK1r+1i.*MSK1i;
    %Ignore any invalid numbers
    MSK1(isnan(MSK1)) = 0;
    %Add the mask to the SLM
    SLM = SLM+MSK1.';
    
end
%View the beam amplitude
figure(1);
imagesc(abs(SLM.'));
axis equal;
%Convert the angle phase to grayscale image
SLM = uint8(mod(((round(256.*((angle(SLM)+pi)./(2.*pi))))),256));

%View the SLM phase mask
figure(2);
imagesc((SLM).');
axis equal
colormap(gray(256));

%Write it to file
imwrite(SLM.','SLM.png');
