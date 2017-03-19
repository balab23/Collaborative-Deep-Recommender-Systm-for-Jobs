function[] =  collab_filtering(num_jobs,num_users,len_features,X,file1)

X = ones(70000,13);
num_jobs=70000;
num_users=60000;
len_features = 13;
save('jobs.mat','X','-v7.3');

i =num_users; %%num of users
j = num_jobs; %%num of jobs
s = len_features; %%num of features

%r = zeros(i,j);
%r = R; %%binary user rating matrix

%R = zeros(i,j);


L = 4; %% number of layers of autoencoder 

K = zeros(L,1); %%vector of num hidden units in each layer
K = [s floor(s/2) s s]; %%num nodes in each hidden layer


mx = matfile('jobs.mat','Writable',true);


%%initialise matrices and vectors

xo = ones(1,1);
x1 = zeros(1,1);
x2 = zeros(1,1);
x3 = zeros(1,1);
x4 = zeros(1,1);
xc = zeros(1,1);
x = X; %%clean vector
xo =x;

save('vars.mat','x1','x2','x3','x4','xc','x','xo' ,'-v7.3');

mv = matfile('vars.mat','Writable',true);



mv.xo =  zeros(s,K(1)); %%noise matrix
mv.x1 = zeros(j,K(1));
mv.x2 = zeros(j,K(2));
mv.x3 = zeros(j,K(3));
mv.x4 = zeros(j,K(4));
mv.xc = zeros(j,K(4));
mv.x = mx.X; %%clean vector
mv.xo =mv.x;



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


save('lambdau','lambdav','-v7.3')

%%-------------------------------------------------

%%initialising extras

u =ones(1,1);
v = ones(1,1);
oldu = ones(1,1);
oldv = ones(1,1);
ut = ones(1,1);
vt = ones(1,1);
in_i=ones(1,1);
in_j=ones(1,1);

save('lvecs.mat','u','v','oldu','oldv','ut','vt','in_i','in_j','-v7.3');
ml = matfile('lvecs.mat','Writable',true);


Ij = K(4);
Ik = K(2);
epsilon = zeros(1,Ik);
ml.v = zeros(j,Ik);
ml.u = zeros(i,Ik);
ml.ut = zeros(1,Ik);
size(ut);
a = 1;
b = 0.01;
l = 0;
%C = [r==1].*a;
%C = C+[C==0].*b;
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

%%--------------------------------------------------- very fast till here tested for 70000*13

for lm=1:1,

for li=1:1,
  
  xi = xo(li,:);
  
  mean = sigmoid(xi*w01+b1');
  variance = 1/lambdas.*eye(K(1));
  mv.x1(li,:) = normalise(mv.x1(li,:),mean,variance);
  
  mean = sigmoid(mv.x1(li,:)*w12+b2');
  variance = 1/lambdas.*eye(K(2));
  mv.x2(li,:) = normalise(mv.x2(li,:),mean,variance);
  
  mean = sigmoid(mv.x2(li,:)*w23+b3');
  variance = 1/lambdas.*eye(K(3));
  mv.x3(li,:) = normalise(mv.x3(li,:),mean,variance);
  
  mean = sigmoid(mv.x3(li,:)*w34+b4');
  variance = 1/lambdas.*eye(K(4));
  mv.x4(li,:) = normalise(mv.x4(li,:),mean,variance);
  
  mean = x4(li,:);
  variance = 1/lambdan.*eye(Ij); %%doubt with Ij
  mv.xc(li,:) = normalise(mv.xc(li,:),mean,variance);
  
  
  mean = 0;
  variance = 1/lambdav.*eye(Ik); %%doubt with Ik
  epsilon = normalise(epsilon,mean,variance);
  
  ml.vt = epsilon + mv.x2(li,:);
  ml.v(li,:)=ml.vt;
  size(vt)
  
  for lk=1:5
  mean = 0;
  variance = 1/lambdav.*eye(Ik);
  ml.ut = normalise(ml.ut,mean,variance);
  size(ut)
  
  ml.u(lk,:) = ml.ut;
  
  ml.in_i=lk;
  ml.in_j=li;
  
  
  % mean = ut*vt' ;
  % size(mean);
  % if(r(lk,li)==1) c = 1/a;
  % else c = 1/b;
  % end
  
  % variance = c;
  % R(lk,li) = normalise(R(lk,li),mean,variance);
  
   commandStr = 'python calc_rating.py';
    [status, commandOut] = system(commandStr);
	if status==0
     fprintf('result succesful %d',lk);
	else fprintf('command string %s %d',commandOut,lk);
    end
  
  end
 end
end %%remove end while running code
 %%---------------------------------------------------------------------------- tested till here extremly slow because of the for loop
  
  
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
   ml.oldu=ml.u;
   ml.oldv=ml.v;
   
   
   
    for li = 1:1,
	ml.in_i = li-1
	commandStr = 'python get_user.py';
    [status, commandOut] = system(commandStr);
	if status==0
     fprintf('result succesful');
	else fprintf('command string %s',commandOut);
    end
   rating = matfile('rating_env_user.mat','Writable',true);
   term = matfile('term.mat','Writable',true);
   R = rating.R;
   R=double(R);
   t = term.t;
   t=t';
   size(t)
   size(ml.v)
   t=double(t);
   ml.u(li,:)=(inv((t*ml.v + lambdau.*eye(K(2))))*(t*R'))';
   end
	
	
	clear R
	clear t
	
   
    for lj = 1:1,
	ml.in_j=lj-1
	commandStr = 'python get_job.py';
    [status, commandOut] = system(commandStr);
	if status==0
    fprintf('result succesful');
	else fprintf('command string %s',commandOut);
    end
    rating = matfile('rating_env_job.mat','Writable',true);
	term = matfile('term.mat','Writable',true);
    R = double(rating.R);
    t = double(term.t);
	t=t';
    v(lj,:)= (inv(t*oldu + lambdav.*eye(K(2)))*(t*R+lambdav.*x2(lj,:)'))';
    end
  
  
%%----------------------------------------------------------------- tested

%% weight gradient

 errorl4 =(mv.x4 -xc).*mv.x4.*(1-mv.x4);
 errorl3 = mv.x3.*(1-mv.x3).*(errorl4*w34');
 errorl22= mv.x2.*(1-mv.x2).*(errorl3*w23');
 errorl2 = (mv.x2 - ml.oldv).*mv.x2.*(1-mv.x2);

 %%layer-1

    term1 = -lambdaw*w01; %l*l
    term2 = -mv.xo'*(lambdav.*mv.x1.*(1-mv.x1).*(errorl2*w12'));
    term3 = -mv.xo'*(lambdan.*mv.x1.*(1-mv.x1).*(errorl22*w12'));
    deltaw01 = term1+term2+term3;


%%layer-2
  
    term1 = -lambdaw.*w12; %%l*l/2
    term2 = -mv.x1'*(lambdav.*(mv.x2 - mv.oldv).*mv.x2.*(1-mv.x2));
    term3 = -mv.x1'*(lambdan.*mv.x2.*(1-mv.x2).*(errorl3*w23'));
    deltaw12 = term1+term2+term3;

%%layer-3

    term1 = -lambdaw.*w23;
    term2 = 0;
    term3 = -mv.x2'*(lambdan.*mv.x3.*(1-mv.x3).*(errorl4*w34'));
    deltaw23 = term1+term2+term3;

%%layer-4
	
    term1 = -lambdaw.*w34;%% l*l
    term2 = 0;
    term3 = -mv.x3'*(lambdan.*(mv.x4 - mv.xc).*mv.x4.*(1-mv.x4));
    deltaw34 = term1+term2+term3;


%% bias calculation.

  term1 = -lambdaw.*b1';
  term2 = -lambdav.*sum(mv.x1.*(1-mv.x1).*(errorl2*w12'));
  term3 = -lambdan.*sum(mv.x1.*(1-mv.x1).*(errorl22*w12'));
  deltab1 = term1+term2+term3; 
  
  term1 = -lambdaw.*b2'; %%l*l/2
  term2 = -lambdav.*sum((mv.x2 - ml.oldv).*mv.x2.*(1-mv.x2));
  term3 = -lambdan.*sum(mv.x2.*(1-mv.x2).*(errorl3*w23'));
  deltab2 = term1+term2+term3;
  
  term1 = -lambdaw.*b3';
  term2 = 0;
  term3 = -lambdan.*sum(mv.x3.*(1-mv.x3).*(errorl4*w34'));
  deltab3 = term1+term2+term3;
  
  term1 = -lambdaw.*b4';%% l*l
  term2 = 0;
  term3 = -lambdan.*sum((mv.x4 - mv.xc).*mv.x4.*(1-mv.x4));
  deltab4 = term1+term2+term3;

  
%%updation
  
w01 = w01 + alpha.*deltaw01;
w12 = w12 + alpha.*deltaw12;
w23 = w23 + alpha.*deltaw23;
w34 = w34 + alpha.*deltaw34;
  
b1 = b1 + alpha.*deltab1';
b2 = b2 + alpha.*deltab2';
b3 = b3 + alpha.*deltab3';
b4 = b4 + alpha.*deltab4';


end

end
