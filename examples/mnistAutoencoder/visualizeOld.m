function visualize(encoderNet, decoderNet, imdb)

if nargin == 0
    clear;
%     opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnist-1-test/net-epoch-5.mat';
%     opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnist-2-test/net-epoch-5.mat';
%     opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnist-3-test/net-epoch-5.mat';

    opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnist-6-test/net-epoch-1.mat';
    nets = load(opts.netFile);
    encoderNet = dagnn.DagNN.loadobj(nets.encoderNet) ;
    decoderNet = dagnn.DagNN.loadobj(nets.decoderNet) ;
    imdb = load('/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnist-1-data/imdb.mat');
end
encoderNet.move('gpu') ;
encoderNet.mode = 'test';
decoderNet.move('gpu') ;
decoderNet.mode = 'test';

%% display filters
filters = encoderNet.params(1).value;
tiledImage = tileImage(gather(filters), 10, 2);
figure;
subplot(2,2,1);
imshow(tiledImage);

%% display some embeddings in a lower dimensional space
n = 512;
sample = randsample(60000, n);
inputs = imdb.images.image(:,:,:,sample);
labels = imdb.images.label(sample);

% embeddingVar = net.getVarIndex('normalizeLp_normL1');
% % embeddingVar = net.getVarIndex('block3_relu');
% net.vars(embeddingVar).precious = true;
encoderNet.eval({'image', gpuArray(inputs)});
% encodings = gather(encoderNet.vars(embeddingVar).value);
encodings = gather(encoderNet.vars(end).value);
encodings = squeeze(encodings);

coefs = pca(encodings);
subplot(2,2,2);
gscatter(coefs(:,1), coefs(:,2), labels-1);

% random points
randomVectors = randn(32,512);
randomVectors = bsxfun(@rdivide, randomVectors, sum(abs(randomVectors),1));
coefs = pca(randomVectors);
subplot(2,2,4);
gscatter(coefs(:,1), coefs(:,2));

% scatter3(coefs(:,1), coefs(:,2), coefs(:,3), [], labels-1);

%% autoencode some samples
sample = randsample(60000, 16);
inputs = imdb.images.image(:,:,:,sample);

reconstructionVar = net.getVarIndex('deblock1_tanh');
% reconstructionVar = net.getVarIndex('block1_a_deconv');
net.vars(reconstructionVar).precious = true;
net.vars(end-1).precious = true;
net.eval({'input', gpuArray(inputs)});
reconstructions = net.vars(reconstructionVar).value;

results = cat(4, inputs, reconstructions) ;

tiledImage = tileImage(gather(results),16,2);
subplot(2,2,3);
imshow(tiledImage);

%% generate some random samples
embeddingLayer = net.getLayerIndex('normalizeLp_normL1');
% embeddingLayer = net.getLayerIndex('block3_abc_deconv');
net.layers = net.layers(embeddingLayer:end);
net.layers(1).inputs = {'input'};
net.rebuild();
% net.removeLayer('repellingloss');
net.removeLayer('loss');

vectorSize = 50;
randomEmbeddings = gpuArray.randn(1,1,vectorSize,256,'single') ;
% randomEmbeddings = sqrt(3)*(2*gpuArray.rand(1,1,vectorSize,256,'single')-1);

net.eval({'input', randomEmbeddings});
reconstructionVar = net.getVarIndex('deblock1_tanh');
% reconstructionVar = net.getVarIndex('block1_a_deconv');
results = gather(net.vars(reconstructionVar).value);
tiledImage = tileImage(results,16,2);
figure;
imshow(tiledImage);

%% interpolation between real samples
sample = randsample(60000, 2);
inputs = imdb.images.image(:,:,:,sample);

results = zeros(16,16,1,256,'single');
embedding2 = encodings(:,1) ;
for i = 1:16
    embedding1 = embedding2 ;
    embedding2 = encodings(:,i) ;

    q = single(linspace(1,0,16));
    interpolatedEmbeddings = embedding1 * q + embedding2 * (1-q);
    interpolatedEmbeddings = gpuArray(permute(interpolatedEmbeddings, [3 4 1 2]));

    net.eval({'input', interpolatedEmbeddings});
    results(:,:,:,(i-1)*16+1:i*16) = gather(net.vars(end).value);
end

tiledImage = tileImage(results,16,2);
figure;
imshow(tiledImage);