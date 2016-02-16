function imdb = setupImdb_mnistMoving2(opts)

load('/esat/malachite/bdebraba/moving-mnist/mnist_test_seq.mat');
data = permute(data, [3,4,1,2]);
% data = reshape(data, 64,64,[]);
% data = imresize(data, 0.5);
% data = permute(data, [1,2,4,3]);
data = single(data);

% binarize the data to 0 and 1
data = round(data / 255);

% rescale between -1 and 1
data = data*2 - 1;

nImages = 9216;

imdb.images.image1 = zeros(64, 64, 1, nImages, 'single') ;
imdb.images.image2 = zeros(64, 64, 1, nImages, 'single') ;
imdb.images.image3 = zeros(64, 64, 1, nImages, 'single') ;
imdb.images.image4 = zeros(64, 64, 1, nImages, 'single') ;

for i = 1:nImages
    imdb.images.image1(:,:,:,i) = data(:,:,1,i) ;
    imdb.images.image2(:,:,:,i) = data(:,:,2,i) ;
    imdb.images.image3(:,:,:,i) = data(:,:,3,i) ;
    imdb.images.image4(:,:,:,i) = data(:,:,4,i) ;
end

% move the ground truth prediction to 1 and 2
imdb.images.image4 = (imdb.images.image4 + 1) / 2 + 1;

imdb.images.set = ones(1,nImages) ;
imdb.images.set(end-1024+1:end) = 2 ;