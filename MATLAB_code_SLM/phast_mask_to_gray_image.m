% === FIXED PHASE MASK EXPORT FOR SLM ===

% After running MPLC_StartHere.m run this to extract gray image for the SLM

% Extract phase of each mask in radians [-pi, pi]
phase1 = angle(squeeze(MASKS(1,:,:)));
phase2 = angle(squeeze(MASKS(2,:,:)));
%phase3 = angle(squeeze(MASKS(3,:,:)));

% Normalize from [-pi, pi] → [0, 1] for grayscale
mask1 = (phase1 + pi) / (2 * pi);
mask2 = (phase2 + pi) / (2 * pi);
%mask3 = (phase3 + pi) / (2 * pi);


% Optional: display both masks to verify
figure;
subplot(1,2,1); imshow(mask1); title('Mask 1 (normalized)');
subplot(1,2,2); imshow(mask2); title('Mask 2 (normalized)');

% Define spacing between masks in pixels (horizontal gap)
%gap = 760;  % → 512 µm on PLUTO (64 * 8 µm)
gap = 90;  % → 512 µm on PLUTO (64 * 8 µm)

% Create white gap (value = 1 → 2π phase)
space = zeros(size(mask1,1), gap);
space_2 = zeros(size(mask1,1), 1160);

% Concatenate: [mask1 | space | mask2]
combinedMask = [mask1, space_2, mask2];
%combinedMask = [mask1, space, mask2, space, mask3];

% Display layout
figure;
imshow(combinedMask);
title('SLM phase layout');

% Optional: scale to 1080×1920 if needed for full SLM screen
% combinedMask_resized = imresize(combinedMask, [1080, 1920]);
% imwrite(combinedMask_resized, 'SLM_PhasePattern.png');

% Save as image file for display/upload
imwrite(combinedMask, 'SLM patterns/l_t_r/2_planes_3modes_800_MFD_test.png');

% Also save as .mat for exact floating-point phase control (optional)
save('SLM_PhasePattern.mat', 'combinedMask');

% Concatenate: [mask1 | space | mask2]
combinedMask = [mask2, space_2, mask1];
%combinedMask = [mask3, space, mask2, space, mask1];


% Save as image file for display/upload
imwrite(combinedMask, 'SLM patterns/r_t_l - r to HG and l to gaussian/2_planes_3modes_800_MFD_test.png');

disp('✅ Phase masks exported successfully for SLM.');
