function visualize(encoderNet, decoderNet, imdb)

if nargin == 0
    clear;
%     opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnist-1-test/net-epoch-5.mat';
%     opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnist-2-test/net-epoch-5.mat';
%     opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnist-3-test/net-epoch-5.mat';

    opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnist-7-test/net-epoch-100.mat';
    nets = load(opts.netFile);
    encoderNet = dagnn.DagNN.loadobj(nets.encoderNet) ;
    decoderNet = dagnn.DagNN.loadobj(nets.decoderNet) ;
    imdb = load('/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnist-7-data/imdb.mat');
end
decoderNet.removeLayer('loss') ;
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

%% autoencode some samples
sample = randsample(60000, 16);
inputs = imdb.images.image(:,:,:,sample);

encoderNet.eval({'image', gpuArray(inputs)});
encodings = encoderNet.vars(end).value ;
encodingsStored = squeeze(encodings) ; % store them for later use.
decoderNet.eval({'encoding', encodings});

reconstructions = decoderNet.vars(end).value;

results = cat(4, inputs, reconstructions) ;

tiledImage = tileImage(gather(results),16,2);
subplot(2,2,3);
imshow(tiledImage);

%% display some embeddings in a lower dimensional space
n = 512;
sample = randsample(60000, n);
inputs = imdb.images.image(:,:,:,sample);
labels = imdb.images.label(sample);

encoderNet.eval({'image', gpuArray(inputs)});
encodings = gather(encoderNet.vars(end).value);
encodings = squeeze(encodings);
% encodings = reshape(inputs, [], n); % visualize the image vectors instead

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

%% generate some random samples
vectorSize = 50;
randomEncodings = gpuArray.randn(1,1,vectorSize,256,'single') ;
% randomEmbeddings = sqrt(3)*(2*gpuArray.rand(1,1,vectorSize,256,'single')-1);

decoderNet.eval({'encoding', randomEncodings});
results = gather(decoderNet.vars(end).value);
tiledImage = tileImage(results,16,2);
figure;
imshow(tiledImage);

%% interpolation between real samples
results = zeros(32,32,1,256,'single');
embedding2 = encodingsStored(:,1) ;
for i = 1:16
    embedding1 = embedding2 ;
    embedding2 = encodingsStored(:,i) ;

    q = single(linspace(1,0,16));
    interpolatedEmbeddings = embedding1 * q + embedding2 * (1-q);
    interpolatedEmbeddings = gpuArray(permute(interpolatedEmbeddings, [3 4 1 2]));

    decoderNet.eval({'encoding', interpolatedEmbeddings});
    results(:,:,:,(i-1)*16+1:i*16) = gather(decoderNet.vars(end).value);
end

tiledImage = tileImage(results,16,2);
figure;
imshow(tiledImage);