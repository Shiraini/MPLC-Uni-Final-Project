function [f] = propagate(f,H0)
    %Convert real-space field, to k-space field
    F = fftshift(fft2(fftshift(f)));
    %Apply the transfer function of free-space
    F = F.*H0;
    %Convert k-space field back to real-space
    f = fftshift(ifft2(fftshift(F)));