function visualize(net)

if nargin == 0
    clear;
    opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnist-5-test/net-epoch-1.mat';
    nets = load(opts.netFile);
    net = dagnn.DagNN.loadobj(nets.net) ;
end
net.move('gpu') ;
net.mode = 'test';

imdb = load('/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnist-5-data/imdb.mat');

%% display filters
filters = net.params(1).value;
tiledImage = tileImage(gather(filters), 10, 2);
figure;
subplot(3,1,1);
imshow(tiledImage);

%% display some embeddings in a lower dimensional space
n = 512;
sample = randsample(60000, n);
inputs = imdb.images.image(:,:,:,sample);
labels = imdb.images.label(sample);

embeddingVar = net.getVarIndex('normalizeL1_normL1');
net.vars(embeddingVar).precious = true;
net.eval({'input', gpuArray(inputs)});
embeddings = gather(net.vars(embeddingVar).value);
embeddings = squeeze(embeddings);

coefs = pca(embeddings);
subplot(3,1,2);
gscatter(coefs(:,1), coefs(:,2), labels-1);
% scatter3(coefs(:,1), coefs(:,2), coefs(:,3), [], labels-1);

%% autoencode some samples
sample = randsample(60000, 16);
inputs = imdb.images.image(:,:,:,sample);

reconstructionVar = net.getVarIndex('deblock1_tanh');
net.vars(reconstructionVar).precious = true;
net.vars(end-1).precious = true;
net.eval({'input', gpuArray(inputs)});
reconstructions = net.vars(reconstructionVar).value;

results = cat(4, inputs, reconstructions) ;

tiledImage = tileImage(gather(results),16,2);
subplot(3,1,3);
imshow(tiledImage);

%% generate some samples
embeddingLayer = net.getLayerIndex('normalizeL1_normL1');
net.layers = net.layers(embeddingLayer:end);
net.layers(1).inputs = {'input'};
net.rebuild();
net.removeLayer('repellingloss');
net.removeLayer('loss');

vectorSize = 128;
randomVectors = gpuArray.randn(1,1,vectorSize,256,'single') ;

net.eval({'input', randomVectors});
reconstructionVar = net.getVarIndex('deblock1_tanh');
results = gather(net.vars(reconstructionVar).value);
tiledImage = tileImage(results,16,2);
figure;
imshow(tiledImage);