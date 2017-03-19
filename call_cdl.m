cd Evaluation/All_Data/

m = matfile('ratings.mat','Writable',true);
R = double(m.R);
m = matfile('jobs.mat','Writable',true);
X = double(m.X);

m = matfile('C:\Users\Admin\Documents\Minor Project\100\ratings_after.mat','Writable',true);
R = double(m.R);
m = matfile('C:\Users\Admin\Documents\Minor Project\100\jobs.mat','Writable',true);
X = double(m.X);



m = matfile('ratings_500.mat','Writable',true);
R = double(m.R);
m = matfile('jobs_500.mat','Writable',true);
X = double(m.X);

u_ids = R(:,1);
R = R(:,2:size(R,2));


num_users = size(R,1); 
num_jobs = size(R,2); 
len_features = size(X,2);


imagesc(R)
xlabel('Jobs')
ylabel('Users')

final_ratings = collab_filtering_direct(num_jobs,num_users,len_features,R,X)

imagesc(final_ratings)
xlabel('Jobs')
ylabel('Users')


m = matfile('ratings_after_500.mat','Writable',true);
R = double(m.R);
m = matfile('jobs_500.mat','Writable',true);
X = double(m.X);

mean_vec = mean(final_ratings,2);
mean_rep = repmat(mean_vec,1,num_jobs);
binary_jobs = [final_ratings >mean_rep]
binary_jobs =[u_ids binary_jobs]

save('recomendations_100_100_c2.mat','binary_jobs')
