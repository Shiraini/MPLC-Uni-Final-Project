function [H0] = transferFunctionOfFreeSpace(x,y,dz,lambda)
s = size(x);
Nx = s(2);
Ny = s(1);

%Setup your k-space co-ordinate system
fs = Nx/((max(max(x))-min(min(x))));
v_x =fs.*([-Nx/2:Nx/2-1]/Nx);
fs = Ny/(max(max(y))-min(min(y)));
v_y =fs.*([-Ny/2:Ny/2-1]/Ny);
[V_x V_y] = meshgrid(v_x,v_y);

%Exponent for the transfer function of free-space
tfCoef1 = -1i.*2.*pi.*sqrt(lambda.^-2-(V_x).^2-V_y.^2);

%Transfer function of free-space for propagation distance dz
H0 = exp(tfCoef1.*dz);
