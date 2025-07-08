function [ex] = HGmodeMFD(X,Y,mfd,l,m)
%Beam waist is MFD/2
W = mfd/2;

%Calculate HG mode
u = sqrt(2).*X./W;
G_l = HermitePoly(l,u).*exp(-u.^2./2);
u = sqrt(2).*Y./W;
G_m = HermitePoly(m,u).*exp(-u.^2./2);
ex = G_l.*G_m;

%Normalize to unit intensity
ex = ex./sqrt(sum(sum(abs(ex).^2)));

%Function for calculating the Hermite polynomials
function y = HermitePoly(n,x)
p(1,1)=1;
if (n>0)
    p(2,1:2)=[2 0];
    if (n>1)
        for k=2:n
            p(k+1,1:k+1)=2*[p(k,1:k) 0]-2*(k-1)*[0 0 p(k-1,1:k-1)];
        end
    end
end
p = p(end,:);

y = polyval(p,x);