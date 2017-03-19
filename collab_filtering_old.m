function[R] =  collab_filtering(num_jobs,num_users,len_features,R,X)

i =num_users; %%num of users
j = num_jobs; %%num of jobs
s = len_features; %%num of features

r = zeros(i,j);
r = R; %%binary user rating matrix

R = zeros(i,j);


L = 4; %% number of layers of autoencoder 

K = zeros(L,1); %%vector of num hidden units in each layer
K = [s s/2 s s]; %%num nodes in each hidden layer


%%initialise matrices and vectors

xo =  zeros(s,K(1)); %%noise matrix
x1 = zeros(j,K(1));
x2 = zeros(j,K(2));
x3 = zeros(j,K(3));
x4 = zeros(j,K(4));
xc = zeros(j,K(4));
x = X; %%clean vector
xo =x;



%%initialising dimensions of weight vectors

w01 =zeros(s,K(1)); %%3X3
w12 =zeros(K(1),K(2));  %%3X1
w23 =zeros(K(2),K(3));  %%1X3
w34 =zeros(K(3),K(4));  %%3X3

%%------------------------------------------------

%%initialise bias vectors

b1 = zeros(K(1),1);  %%3X1
b2 = zeros(K(2),1);  %%1X1
b3 = zeros(K(3),1);  %%3X1
b4 = zeros(K(4),1);  %%3X1

%%-------------------------------------------------




%%hyperparameters initialisation // need to perform grid search

lambdaw = 0.01;
lambdan = 0.01;
lambdau = 0.01;
lambdas = 0.01;
lambdav = 0.01;

%%-------------------------------------------------

%%initialising extras

Ij = K(4);
Ik = K(2);
epsilon = zeros(Ik,1);
v = zeros(j,Ik);
u = zeros(i,Ik);
ut = size(Ik,1);
a = 1;
b = 0.01;
l = 0;
C = [r==1].*a;
C = C+[C==0].*b;
alpha = 1;

%%-----------------------------------------------------------

%%initialise weights and bias vectors


mean = 0;

variance = 1/lambdaw.*eye(K(1));
w01 = normalise(w01,mean,variance);
b1 = normalise(b1,mean,variance);

variance = 1/lambdaw.*eye(K(2));
w12 = normalise(w12,mean,variance);
b2 = normalise(b2,mean,variance);

variance = 1/lambdaw.*eye(K(3));
w23 = normalise(w23,mean,variance);
b3 = normalise(b3,mean,variance);

variance = 1/lambdaw.*eye(K(4));
w34 = normalise(w34,mean,variance);
b4 = normalise(b4,mean,variance);

%%---------------------------------------------------

for lm=1:1000,

for li=1:j,
  
  xi = xo(li,:);
  
  mean = sigmoid(xi*w01+b1');
  variance = 1/lambdas.*eye(K(1));
  x1(li,:) = normalise(x1(li,:),mean,variance);
  
  mean = sigmoid(x1(li,:)*w12+b2');
  variance = 1/lambdas.*eye(K(2));
  x2(li,:) = normalise(x2(li,:),mean,variance);
  
  mean = sigmoid(x2(li,:)*w23+b3');
  variance = 1/lambdas.*eye(K(3));
  x3(li,:) = normalise(x3(li,:),mean,variance);
  
  mean = sigmoid(x3(li,:)*w34+b4');
  variance = 1/lambdas.*eye(K(4));
  x4(li,:) = normalise(x4(li,:),mean,variance);
  
  mean = x4(li,:);
  variance = 1/lambdan.*eye(Ij); %%doubt with Ij
  xc(li,:) = normalise(xc(li,:),mean,variance);
  
  
  mean = 0;
  variance = 1/lambdav.*eye(Ik); %%doubt with Ik
  epsilon = normalise(epsilon,mean,variance);
  
  vt = epsilon' + x2(li,:);
  v(li,:)=vt;
 
  
  for lk=1:i
  mean = 0;
  variance = 1/lambdav.*eye(Ik);
  ut = normalise(ut,mean,variance);
  
  u(lk,:) = ut;
  
  mean = u'*v ;
  if(r(lk,li)==1) c = 1/a;
  else c = 1/b;
  
  end
  variance = c;
  R(lk,li) = normalise(R(lk,li),mean,variance);
  
  end

  end
 %%---------------------------------------------------------------------------- tested
  
  
 %% calculating L
 
%   sumu=0;
%   sumv=0;
%   sumx=0;
%   sumc=0;
%   
%   for li=1:i,
%    sumu = sumu + norm(u(i,:))^2;  
%  end
%  
%   for lj = 1:j,
%     sumv= sumv + norm(v(li,:) - x2(li,:))^2;
% 	sumx= sumx + norm(xc(li,:) - x(li,:))^2; 
%     
%     for li = 1:i,
% 	   if(R(li,lj)==1) c = a;
% 	   else c = b;
%        sumc = sumc + c/2*(R(li,lj)-u(li,:)'*v(lj,:))^2; 
% 	end
% 	
%  end   
%   
%    term1 = -lambdau/2*sumu;
%    term2 = -lambdaw/2*(norm(w01,'fro')^2 + norm(w12,'fro')^2 + norm(w23,'fro')^2 + norm(w34,'fro')^2 + norm(b1)^2 +norm(b2)^2 +norm(b3)^2+norm(b4)^2);
%    term3 = -lambdav/2*sumv;
%    term4 = -lambdan/2*sumx;
%    term5 = -sumc
   
   
%%----------------------------------------------------------------  
   oldu=u;
   oldv=v;
   
   for li = 1:i,
   u(li,:)=(inv(v'*diag(C(li,:))*v + lambdau*eye(K(2)))*v'*diag(C(li,:))*R(li,:)')'
   end
   
   for lj = 1:j,
   v(lj,:)= (inv(oldu'*diag(C(:,lj))*oldu + lambdav*eye(K(2)))*(oldu'*diag(C(:,lj))*R(:,lj)+lambdav.*x2(lj,:)'))'
   end
  
  
%%----------------------------------------------------------------- tested

%% weight gradient

 errorl4 =(x4 -xc).*x4.*(1-x4);
 errorl3 = x3.*(1-x3).*(errorl4*w34')
 errorl22= x2.*(1-x2).*(errorl3*w23')
 errorl2 = (x2 - oldv).*x2.*(1-x2);

 %%layer-1

    term1 = -lambdaw*w01; %l*l
    term2 = -xo'*lambdav.*x1.*(1-x1).*(errorl2*w12')
    term3 = -xo'*lambdan.*x1.*(1-x1).*(errorl22*w12')
    deltaw01 = term1+term2+term3;


%%layer-2
  
    term1 = -lambdaw.*w12; %%l*l/2
    term2 = -x1'*(lambdav.*(x2 - oldv).*x2.*(1-x2));
    term3 = -x1'*(lambdan.*x2.*(1-x2).*(errorl3*w23'))
    deltaw12 = term1+term2+term3;

%%layer-3

    term1 = -lambdaw.*w23;
    term2 = 0;
    term3 = -x2'*(lambdan.*x3.*(1-x3).*(errorl4*w34'))
    deltaw23 = term1+term2+term3;

%%layer-4
	
    term1 = -lambdaw.*w34;%% l*l
    term2 = 0;
    term3 = -x3'*(lambdan.*(x4 - xc).*x4.*(1-x4));
    deltaw34 = term1+term2+term3;

w01 = w01 + alpha.*deltaw01;
w12 = w12 + alpha.*deltaw12;
w23 = w23 + alpha.*deltaw23;
w34 = w34 + alpha.*deltaw34;

%% bias calculation.

  term1 = -lambdaw.*b1';
  term2 = -lambdav.*sum(x1.*(1-x1).*(errorl2*w12'))
  term3 = -lambdan.*sum(x1.*(1-x1).*(errorl22*w12'))
  deltab1 = term1+term2+term3; 
  
  term1 = -lambdaw.*b2'; %%l*l/2
  term2 = -lambdav.*sum((x2 - oldv).*x2.*(1-x2));
  term3 = -lambdan.*sum(x2.*(1-x2).*(errorl3*w23'))
  deltab2 = term1+term2+term3;
  
  term1 = -lambdaw.*b3';
  term2 = 0;
  term3 = -lambdan.*sum(x3.*(1-x3).*(errorl4*w34'))
  deltab3 = term1+term2+term3;
  
  term1 = -lambdaw.*b4';%% l*l
  term2 = 0;
  term3 = -lambdan.*sum((x4 - xc).*x4.*(1-x4));
  deltab4 = term1+term2+term3;

b1 = b1 + alpha.*deltab1';
b2 = b2 + alpha.*deltab2';
b3 = b3 + alpha.*deltab3';
b4 = b4 + alpha.*deltab4';


end

end
