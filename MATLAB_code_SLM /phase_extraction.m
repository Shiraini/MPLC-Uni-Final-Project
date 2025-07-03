load('PhaseMasks.mat'); % MASKS

phase_masks = angle(MASKS); % extracting from each compex number onlt the phase

SLM_res_x = 1920;
SLM_res_y = 1080;

for idx = 1:size(phase_masks, 1)
    mask = phase_masks(idx, :, :);
    mask = squeeze(mask); % reducing non relevant dimensions
    
    mask_normalized = mod(mask, 2*pi); % change the range to be from -pi to pi
    mask_SLM = uint8(mask_normalized / (2*pi) * 255); % 8 bits convertion
    
    mask_resized = imresize(mask_SLM, [SLM_res_y, SLM_res_x], 'nearest');
    
    filename = sprintf('PhaseMask_plane_%d.png', idx);
    imwrite(mask_resized, filename);
end
