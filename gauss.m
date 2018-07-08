function p = Gauss(x,mu,sig)

normalise = 1./sqrt(det(sig)*(2*pi)^3);
expon = exp(-0.5*(x-mu)'*inv(sig)*(x-mu));
p = normalise * expon;

end