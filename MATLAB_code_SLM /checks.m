% Load the phase masks
MASKS_4_struct = load('4_PhaseMasks.mat', 'MASKS');
MASKS_8_struct = load('8_PhaseMasks.mat', 'MASKS');

% Extract the MASKS variables from the structures
MASKS_4 = MASKS_4_struct.MASKS;
MASKS_8 = MASKS_8_struct.MASKS;

% Check size
size_4 = size(MASKS_4);
size_8 = size(MASKS_8);

disp(['Size of MASKS_4: ', mat2str(size_4)]);
disp(['Size of MASKS_8: ', mat2str(size_8)]);

% Check differences
if isequal(size_4, size_8)
    % Compute norm of the difference if sizes are the same
    diff = norm(MASKS_4(:) - MASKS_8(:)); 
    disp(['Difference norm: ', num2str(diff)]);
else
    disp('MASK sizes do not match.');
end
