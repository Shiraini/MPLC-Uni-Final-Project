function [MODES,M,N,TOTAL] = generateBasisHG(maxMG,X,Y,mfd)
%Create a basis of HG modes up to mode-group maxMG
maxMG = abs(maxMG);
%Setup the pixel dimensions given the X-coordinate system specified
s = size(X);
Nx = s(1);
Ny = s(2);

%Total number of modes to generate
modeCount = sum(1:maxMG);

%M and N indicies for each mode
M = zeros(1,modeCount,'single');
N = zeros(1,modeCount,'single');

%Field for each mode
MODES = zeros(modeCount,Nx,Ny,'single');
%Summary. Sum of the total intensity of all modes
TOTAL = zeros(Nx,Ny,'single');
%Current mode index
idx=1;
%for each mode-group
for mgIdx=1:maxMG
    %zero-based index of the mode-group
    mgIDX = mgIdx-1;
    %For every mode in this group (there will be mgIdx of them)
    for modeIdx=1:mgIdx
        %m+n should equal mgIDX.
        %Go through each m,n combo in this group starting with max m
        m = mgIDX-(modeIdx-1);
        n = mgIDX-m;
        %Calculate this HG(m,n) mode
        [MODE] = HGmodeMFD(X,Y,mfd,m,n);
        %Store the resulting field
        MODES(idx,:,:) = MODE;
        %Store the m,n index of this mode
        M(idx) = m;
        N(idx) = n;
        %Add the intensity of this mode to the summary intensity
        TOTAL = TOTAL+abs(MODE).^2;
        %Increment mode index
        idx=idx+1;
    end
end
