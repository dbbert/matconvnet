function imdb = setupImdb_mnistMoving(opts)

load('/esat/malachite/bdebraba/moving-mnist/mnist_test_seq.mat');
data = permute(data, [3,4,1,2]);
data = reshape(data, 64,64,[]);
% data = imresize(data, 0.5);
data = permute(data, [1,2,4,3]);
data = single(data);

% rescale between -1 and 1
data = (data / 255)*2 - 1;

% % resize to 16x16
% data = permute(data, [1,2,4,3]);
% data = imresize(data, 0.5);
% data = permute(data, [1,2,4,3]);

% nImages = size(data,4) ;
nImages = 199680; % divisible by 1024
imdb.images.image = data(:,:,:,1:nImages) ;
imdb.images.label = ones(1,nImages) ;
imdb.images.set = ones(1,nImages) ;
imdb.images.set(end-10240+1:end) = 2 ;
% imdb.images.set(9*end/10+1:end) = 2 ;