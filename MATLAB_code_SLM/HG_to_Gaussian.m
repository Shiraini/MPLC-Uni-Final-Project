clear all;
close all;

% === SETTINGS ===
viewMaskFromPaper = 0;
symmetricMasks = 1;

maxMG = 2;
modeCount = sum(1:maxMG);  % e.g. HG00, HG10, HG01 = 3 modes

lambda = 632.8e-9;
downsample = 1;
pixelSize = downsample * 8e-6;

planeSpacing = 92.144e-3;
arrayDistToFirstPlane = 49.525e-3;
planeCount = 3;
iterationCount = 100;
graphIterations = 20;

MFDin = 800e-6;     % HG input modes
MFDout = 40e-6;     % Gaussian spot output

kSpaceFilter = 1;

Nx = 1000;
Ny = 640;
maskOffset = sqrt(1e-3./(Nx.*Ny.*modeCount));


% === COORDINATE GRIDS ===
X = ((1:Ny)-(Ny/2+0.5)) * pixelSize;
Y = ((1:Nx)-(Nx/2+0.5)) * pixelSize;
[X, Y] = meshgrid(X, Y);
[TH, R] = cart2pol(X, Y);
[X0, Y0] = pol2cart(TH - pi/4, R);  % Rotate for HG modes

% === INPUT HG MODES ===
disp('Generating input HG modes...');
[MODES, M, N, ~] = generateBasisHG(maxMG, X0, Y0, MFDin);

% === OUTPUT SPOTS from ARRAY_465a.mat ===
disp('Generating target Gaussian spots from array...');
load('ARRAY_465a.mat');  % Contains x, y
x = x(1:modeCount);
y = y(1:modeCount);
Z = ones(size(X)) * arrayDistToFirstPlane;
[SPOTS, ~] = fibreArrayXYZ(x, y, Z, X, Y, MFDout, lambda);

% === PLOT SPOTS TO CONFIRM ===
figure(10); clf;
for i = 1:modeCount
    subplot(1, modeCount, i);
    imagesc(abs(squeeze(SPOTS(i,:,:))).^2);
    axis image off;
    title(sprintf('Spot %d', i));
end

% === PLOT INPUT MODES TO CONFIRM ===
figure(11); clf;
for i = 1:modeCount
    subplot(1, modeCount, i);
    imagesc(abs(squeeze(MODES(i,:,:))).^2);
    axis image off;
    title(['HG(' num2str(M(i)) ',' num2str(N(i)) ')']);
end

% === INITIALIZATION ===
FIELDS = zeros(2, planeCount, modeCount, Nx, Ny, 'single');
MASKS = ones(planeCount, Nx, Ny, 'single');
coupling = zeros(iterationCount, modeCount);

FIELDS(1,1,:,:,:) = MODES;
FIELDS(2,planeCount,:,:,:) = SPOTS;

% === TRANSFER FUNCTION ===
H0 = transferFunctionOfFreeSpace(X, Y, planeSpacing, lambda);
maxR = max(R(:));
H = H0 .* (R < kSpaceFilter * maxR);

% === INITIAL PROPAGATION ===
for directionIdx = [1 2]
    if directionIdx == 1
        h = H; pRange = 1:(planeCount-1);
        maskSign = -1i;
    else
        h = conj(H); pRange = planeCount:-1:2;
        maskSign = 1i;
    end
    for planeIdx = pRange
        MASK = exp(maskSign * angle(squeeze(MASKS(planeIdx,:,:))));
        for modeIdx = 1:modeCount
            field = squeeze(FIELDS(directionIdx,planeIdx,modeIdx,:,:));
            field = field .* MASK;
            field = propagate(field, h);
            FIELDS(directionIdx,planeIdx + sign(diff(pRange)),modeIdx,:,:) = field;
        end
    end
end

% === ITERATIONS ===
for i = 1:iterationCount
    % Forward
    h = H; directionIdx = 1;
    for planeIdx = 1:(planeCount-1)
        if ~viewMaskFromPaper
            updateMask
        end
        MASK = exp(-1i * angle(squeeze(MASKS(planeIdx,:,:))));
        for modeIdx = 1:modeCount
            field = squeeze(FIELDS(directionIdx,planeIdx,modeIdx,:,:));
            field = field .* MASK;
            field = propagate(field, h);
            FIELDS(directionIdx,planeIdx+1,modeIdx,:,:) = field;
        end
    end

    % Backward
    h = conj(H); directionIdx = 2;
    for planeIdx = planeCount:-1:2
        if ~viewMaskFromPaper
            updateMask
        end
        MASK = exp(1i * angle(squeeze(MASKS(planeIdx,:,:))));
        for modeIdx = 1:modeCount
            field = squeeze(FIELDS(directionIdx,planeIdx,modeIdx,:,:));
            field = field .* MASK;
            field = propagate(field, h);
            FIELDS(directionIdx,planeIdx-1,modeIdx,:,:) = field;
        end
    end

    if mod(i, graphIterations) == 0
        fprintf('Iteration %d/%d\n', i, iterationCount);
        graphFields;
        drawnow;
    end
end

% === FINAL COUPLING CALCULATION ===
h = H0;
for planeIdx = 1:(planeCount-1)
    MASK = exp(-1i * angle(squeeze(MASKS(planeIdx,:,:))));
    for modeIdx = 1:modeCount
        field = squeeze(FIELDS(1,planeIdx,modeIdx,:,:));
        field = field .* MASK;
        field = propagate(field, h);
        FIELDS(1,planeIdx+1,modeIdx,:,:) = field;
    end
end

couplingMatrix = zeros(modeCount, modeCount);
for modeIdx = 1:modeCount
    fieldIn = conj(squeeze(FIELDS(1,planeCount,modeIdx,:,:)));
    fieldIn = fieldIn .* exp(1i * angle(squeeze(MASKS(planeCount,:,:))));
    for modeIdy = 1:modeCount
        fieldOut = squeeze(FIELDS(2,planeCount,modeIdy,:,:));
        couplingMatrix(modeIdx,modeIdy) = sum(sum(fieldIn .* fieldOut));
    end
end

figure(12); clf;
imagesc(abs(couplingMatrix)); axis image; colorbar;
xlabel('Spot index'); ylabel('HG mode index');
title('Coupling Matrix');

[~, S, ~] = svd(couplingMatrix);
s = diag(S).^2;
IL = 10 * log10(mean(s));
MDL = 10 * log10(max(s)/min(s));
fprintf('Insertion Loss (IL): %.2f dB\n', IL);
fprintf('Mode Dependent Loss (MDL): %.2f dB\n', MDL);

save('PhaseMasks.mat', 'MASKS');

% === SHOW FINAL FIELD FOR EACH MODE ===
figure(13); clf;
for i = 1:modeCount
    subplot(1, modeCount, i);
    imagesc(abs(squeeze(FIELDS(1,planeCount,i,:,:))).^2);
    axis image off;
    title(['HG mode ' num2str(i) ' → Output']);
end
