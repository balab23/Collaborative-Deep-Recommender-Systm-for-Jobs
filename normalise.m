function[values] = normalise(x,mean,variance)

sd = sqrt(variance);

w = zeros(size(x));

sd = norm(sd);
size(sd)
size(mean)

w = mean + randn(size(w,1),size(w,2)).*sd;

values = w;

end