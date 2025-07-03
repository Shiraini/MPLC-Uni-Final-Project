clear all;
%close all;

%Settings this flag will view the 210 mode mask from the paper. No mask
%updating. Setting this flag to zero will start calculating a new mask.
%viewMaskFromPaper = 1;
viewMaskFromPaper = 0;

%total mode groups
maxMG = 2;
modeCount = sum(1:maxMG);

%Centre Wavelength
lambda = 632.8e-9;

%Downsample the simulations N times from full-res
downsample = 1;

%Phase mask pixel pitch
pixelSize = downsample.*8e-6;

%Plane spacing
planeSpacing = 92.144e-3;

%How far is the input SMF array from the first plane?
arrayDistToFirstPlane = 49.525e-3;

%Total number of planes
% The less planes the better
planeCount = 2;

%Pixel counts of the masks and simulation in x and y dimensions
if (viewMaskFromPaper)
    Nx = 512./downsample;
    Ny = 384./downsample;
else
    %Nx = 512./downsample;
    %Ny = 448./downsample;
    Nx = 1000./downsample;
    Ny = 380./downsample;
end

%number of total passes to run for
if (viewMaskFromPaper)
    iterationCount = 1;
else
    iterationCount = 100;
end
%Display graphs of fields/masks every N passes
graphIterations = 10;

%Mode-field diameter (MFD) of input Gaussian beams
MFDin = 60e-6;

%Mode-field diameter (MFD) of the output Hermite-Gaussian beams
MFDout = 800e-6;

%SIMULATION CONSTRAINTS
%Angle-space filter. Only propagate angles less than kSpaceFilter*(the
%maximum angle supported by the simulation. Which would in turn be given by
%the pixel size)
%e.g. 0.5 means only half the possible angle-space is used. Limiting angle
%space encourages solutions which are high-bandwidth, have low scatter,
%smooth and don't accidently wrap-around the edge of the simulation.
kSpaceFilter = 1000;

%A small offset that is added to the mask just before the phase-only is
%taken. This discourages the simulation from phase-matching low-intensity
%parts of the field, and encourages solutions which are higher-bandwidth,
%smoother and with less scatter. The Nx.*Ny.*modeCount normalization tries
%to keep the value consistent even if the resolution or number of modes is
%changed.
maskOffset = sqrt(1e-3./(Nx.*Ny.*modeCount));
%Force the masks to be symmetric (top half is the same as bottom half of
%mask)
symmetricMasks = 1;

%Setup mask Cartesian co-ordinates/
%0.5 pixel offset makes the problem symmetric in x and y
X = ((1:Ny)-(Ny./2+0.5)).*pixelSize;
Y = ((1:Nx)-(Nx./2+0.5)).*pixelSize;
[X Y] = meshgrid(X,Y);

%Create the HG-modes
disp('Generating output basis...');
%Convert to polar-coordinates and rotate by 45degrees
[TH R] = cart2pol(X,Y);
[X0 Y0] = pol2cart(TH-pi/4,R);
%Generate a set of HG modes up to mode-group maxMG
[MODES,M,N, MODES_TOTAL] = generateBasisHG(maxMG,X0,Y0,MFDout);

%Create the SMF spot array
disp('Generating input basis...');
%Load the x,y coordinates of each spot from file
load('ARRAY_465a.mat');
%Just take the number of spots we're using in this simulation
x = x(1:modeCount);
y = y(1:modeCount);
%array specifing the z-axis (offset to the first plane)
Z = ones(size(X)).*arrayDistToFirstPlane;
%Calculate all spots at positions (x,y) using the co-ordinate system
%(X,Y,Z), at the specified wavelength (lambda)
[SPOTS SPOTS_TOTAL] = fibreArrayXYZ(x,y,Z,X,Y,MFDin,lambda);

%Print-out how much memory you're going to need for all the fields (every
%mode at every plane, in both directions).
memoryRequiredGB = (2.*planeCount.*modeCount.*Nx.*Ny.*8)./(1024.^3);
fprintf('This simulation requires over %3.3f GB of RAM\n',memoryRequiredGB);

%Allocate all the fields (both directions, every plane, every mode, pixels
%x ,pixels y)
FIELDS = zeros(2,planeCount,modeCount,Nx,Ny,'single');

%Allocate all masks. Here set to blank phase, but these could be any
%initial state you wish. e.g. lens-like masks
MASKS = ones(planeCount,Nx,Ny,'single');
%If we're viewing the mask from the paper. Load that mask set, and
%interpolate it to the current resolution (if using downsampling)
if (viewMaskFromPaper)
    load('..\PhaseMasks\210\HG210_PLUTOII.mat','MASKS');
    MASKS = conj(MASKS);
    s = size(MASKS);
    maskCount = s(1);

    MASKS0 = zeros(s(1),s(3),s(2),'single');
    for maskIdx=1:maskCount
        MASK0(maskIdx,:,:) = squeeze(MASKS(maskIdx,:,:)).';
    end
    MASKS = MASK0;
    if (downsample>1)
        s = size(MASKS);
        maskCount = s(1);
        MASKS0 = zeros(s(1),Nx,Ny,'single');
        for maskIdx=1:maskCount
            MASK = squeeze(MASKS(maskIdx,:,:));
            MASKr = real(MASK);
            MASKi = imag(MASK);
            MASKr = imresize(MASKr,1./downsample);
            MASKi = imresize(MASKi,1./downsample);
            MASKS0(maskIdx,:,:) = MASKr+1i.*MASKi;
        end
        MASKS = MASKS0;
    end
end
%the coupling between the input and output modes. Only approximate once a
%kSpaceFilter<1 is used as not all power is propagated from plane-to-plane in
%that case.
coupling = zeros(iterationCount,modeCount);

%The transfer function of free-space. This is used to propagate from
%plane-to-plane
%size(lambda)
H0 = transferFunctionOfFreeSpace(X,Y,planeSpacing,lambda);
%Filter the transfer function. Removing any k-components higher than
%kSpaceFilter*k_max.
maxR = max(max(R));
H = H0.*(R<(kSpaceFilter.*maxR));

%Initialise fields
%Put the spots as the field in the first plane travelling forward
FIELDS(1,1,:,:,:) = SPOTS;
%Put the HG-modes as the field in the last plane travelling backward
FIELDS(2,planeCount,:,:,:) = MODES;

%Setup the initial fields at each plane in each direction...

%Setup forward propagation
%Index used to pick off the FIELD in this direction
directionIdx = 1;
%Transfer function of free-space (H)
h = H;
for planeIdx=1:(planeCount-1)
    %Conjugate of the mask in this plane. Whether to conjugate or not just
    %depends on how you've set up all the conjugates, fields and overlaps
    %throughout the simulation. Main thing is to be consistent throughout.
    MASK = exp(-1i.*angle(squeeze(MASKS(planeIdx,:,:))));
    %For every mode.
    for modeIdx=1:modeCount
        %Get the field of this mode in this plane
        field = squeeze(FIELDS(directionIdx,planeIdx,modeIdx,:,:));
        %Apply the mask
        field = field.*MASK;
        %Propagate it to the next plane
        field = propagate(field,h);
        %Store the result
        FIELDS(directionIdx,planeIdx+1,modeIdx,:,:) = field;
    end
end

%Setup backwards field
%Index used to pick off the FIELD in this direction
directionIdx = 2;
%Travelling backwards so the transfer function of free-space is conjugate
%(-z). Again, what is and isn't conjugated just depends on the conventions
%you've chosen, but will have to be consistent throughout for it to work.
h = conj(H);
for planeIdx=planeCount:-1:2
    %The phase of the mask in this plane
    MASK = exp(1i.*angle(squeeze(MASKS(planeIdx,:,:))));
    %For every mode in this direction
    for modeIdx=1:modeCount
        field = squeeze(FIELDS(directionIdx,planeIdx,modeIdx,:,:));
        %Apply the mask
        field = field.*MASK;
        %Propagate backwards to the previous plane
        field = propagate(field,h);
        %Store the result
        FIELDS(directionIdx,planeIdx-1,modeIdx,:,:) = field;
    end
end
%All the fields are initialised now. Technically we didn't need to setup
%the forward direction as we'll be starting from the first plane and 
%re-calculating that in the first iteration.

%Time to iterate through and update the masks so attempt to phase-match all
%modes propagating in both directions.
for i=1:iterationCount
    
    %Propagate from first plane to last plane
    h = H;
    directionIdx=1;
    for planeIdx=1:(planeCount-1)
        %Update the mask (see seperate script updateMask.m)
        if (~viewMaskFromPaper)
            updateMask
        end
        %Take the conjugate phase of the mask in this plane
        MASK = exp(-1i.*angle(squeeze(MASKS(planeIdx,:,:))));
        
        %For every mode in the forward direction
        for modeIdx=1:modeCount
            field = squeeze(FIELDS(directionIdx,planeIdx,modeIdx,:,:));
            %Apply the mask
            field = field.*MASK;
            %Progagate to the next plane
            field = propagate(field,h);
            %Store the result
            FIELDS(directionIdx,planeIdx+1,modeIdx,:,:) = field;
        end
    end
    
    %Propagate backwards from last plane to first plane
    %Propagating backwards so conjugate transfer function
    h = conj(H);
    directionIdx=2;
    for planeIdx=planeCount:-1:2
        %Update the mask (see seperate script updateMask.m)
        if (~viewMaskFromPaper)
            updateMask
        end
        %the phase of the mask for this plane
        MASK = exp(1i.*angle(squeeze(MASKS(planeIdx,:,:))));
        %For every mode...
        for modeIdx=1:modeCount
            field = (squeeze(FIELDS(directionIdx,planeIdx,modeIdx,:,:)));
            %Apply the mask
            field = field.*(MASK);
            %Propagate backwards to previous plane
            field = propagate(field,h);
            %Store the result
            FIELDS(directionIdx,planeIdx-1,modeIdx,:,:) = (field);
        end  
    end
    %Graph the fields and masks every graphIterations passes
    if (mod(i,graphIterations)==0)
        graphFields;
    end
     
    %Plot the input/output couplings as the simulation progresses
    figure(3);
    plot(1:i,10.*log10(coupling(1:i,:)));
    set(gca,'YLim',[-20 0]);
    xlabel('Pass');
    ylabel('Coupling (dB)');
end

%Final error calculation

%Use the un-filtered transfer function of free-space (the entire
%angle-space)
h = H0;
%Propagate from the first plane to the last plane in the forward direction
directionIdx = 1;
for planeIdx=1:(planeCount-1)
    %Conjugate phase of the mask in this plane.
    MASK = exp(-1i.*angle(squeeze(MASKS(planeIdx,:,:))));
    %For every mode
    for modeIdx=1:modeCount
        field = squeeze(FIELDS(directionIdx,planeIdx,modeIdx,:,:));
        %Apply mask
        field = field.*MASK;
        %Propagate to next plane
        field = propagate(field,h);
        %Store the result
        FIELDS(directionIdx,planeIdx+1,modeIdx,:,:) = field;
    end
end

%Matrix of coupling every mode in, to every mode out
couplingMatrix = zeros(modeCount,modeCount);

%For every mode in
for modeIdx=1:modeCount
    %get the conjugate field in the forward direction at the last plane
    fieldIn = conj(squeeze(FIELDS(1,planeCount,modeIdx,:,:)));
    %Apply the phase mask to the field
    fieldIn = fieldIn.*squeeze(exp(1i.*angle(MASKS(planeCount,:,:))));
    %For every mode out
    for modeIdy=1:modeCount
        %get the field in the backward direction at the last plane
        fieldOut = squeeze(FIELDS(2,planeCount,modeIdy,:,:));
        %Calculate the overlap integral between the two (fieldIn already
        %conjugated)
        couplingMatrix(modeIdx,modeIdy) = sum(sum(fieldIn.*fieldOut));
    end
end

%Plot the amplitude of the coupling matrix
figure(3);
imagesc(abs(couplingMatrix));
axis equal;

%Perform the singular-value decomposition (SVD) of the coupling matrix to
%calculate the insertion loss (IL) and mode-dependent loss (MDL)
[U S V] = svd(couplingMatrix);
%singular values squared
s = diag(S).^2;
%Insertion loss is the mean singular value squared
IL = 10.*log10(mean(s))
%Mode-dependent loss is the ratio between the maximum and minimum singular
%value squared
MDL = 10.*log10(s(end)./s(1))

%Mask mat for the interpolation
save('PhaseMasks.mat', 'MASKS');

%Done. Happy moding.