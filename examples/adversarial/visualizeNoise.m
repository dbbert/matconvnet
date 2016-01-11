function visualizeNoise(generatorNet)

if nargin == 0
    close all;
    opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/adversarial/dataNoise/mnist-4-test/net-epoch-18.mat';
    % opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/adversarial/dataNoise/mnist-2-test/net-epoch-50.mat';
    % opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/adversarial/dataNoise/mnist-1-test/net-epoch-50.mat';
    % opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/adversarial/data/toyotaThumbs-1-test/net-epoch-11.mat';

    nets = load(opts.netFile);
    generatorNet = dagnn.DagNN.loadobj(nets.generatorNet) ;
end

generatorNet.move('gpu') ;
generatorNet.mode = 'test';

imdb = load('/users/visics/bdebraba/devel/matconvnet/examples/adversarial/dataNoise/mnist-3-data/imdb.mat');
% A = find(ismember(imdb.images.label, [1,2,3,4]));
A = find(ismember(imdb.images.label, [1:10]));

%% display filters
filters = generatorNet.params(1).value;
tiledImage = tileImage(gather(filters), 10, 2);
figure;
subplot(2,1,1);
imshow(tiledImage);

%% display n mnist characters and the distance matrix
nImages = numel(imdb.images.set);
n = 512;

sample = A(randsample(numel(A), n));
randomVectors = imdb.images.image(:,:,:,sample);
labels = imdb.images.label(sample);

% mnistOnes = imdb.images.image(:,:,:,imdb.images.label == 2);
% mnistTwos = imdb.images.image(:,:,:,imdb.images.label == 3);
% mnistThrees = imdb.images.image(:,:,:,imdb.images.label == 4);
% randomVectors = cat(4, mnistOnes(:,:,:,1:5), mnistTwos(:,:,:,1:5), mnistThrees(:,:,:,1:5));

% tiledImage = tileImage(randomVectors,16,2);
% figure;
% imshow(tiledImage);

generatorNet.eval({'input', gpuArray(randomVectors)});
featureVectors = gather(generatorNet.vars(end).value);
featureVectors = squeeze(featureVectors);
% 
% distMatrix = dist(featureVectors);
% 
% figure;
% imagesc(distMatrix);
% colorbar;

%% plot them in a lower dimensional space
coefs = pca(featureVectors);
subplot(2,1,2);
gscatter(coefs(:,1), coefs(:,2), labels-1);
% scatter3(coefs(:,1), coefs(:,2), coefs(:,3), [], labels-1);