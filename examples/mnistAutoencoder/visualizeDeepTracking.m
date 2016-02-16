function visualizeDeepTracking(net, imdb)

if nargin == 0
    clear;
    imdb = load('/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnistMoving2-1-data/imdb.mat');    
%     opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnistMoving2-1-test/net-epoch-3.mat';
    opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnistMoving2-2-test/net-epoch-5.mat';
    
    net = load(opts.netFile);
    net = dagnn.DagNN.loadobj(net.net) ;
    
    net.removeLayer(net.layers(end).name);
    net.addLayer('prob', dagnn.SoftMax(), net.layers(end).name, 'prob', {}) ;
    net.rebuild();
end
% decoderNet.removeLayer('loss') ;
net.move('gpu') ;
encoderNet.mode = 'test';
% net.mode = 'normal';

%% display filters
filters = net.params(1).value;
tiledImage = tileImage(gather(filters), 10, 2);
figure;
subplot(2,2,1);
imshow(tiledImage);

%% display some random future predictions

nSamples = 8;
nImages = numel(imdb.images.set);
rng(3);
sample = randsample(nImages, nSamples);

inputs = getDagNNBatch(imdb, sample);

net.eval(inputs);

predictions = squeeze(net.vars(end).value) ;

for i = 1:nSamples
    subplot(nSamples,5,(i-1)*5+1);
    imshow(inputs{2}(:,:,:,i));
    subplot(nSamples,5,(i-1)*5+2);
    imshow(inputs{4}(:,:,:,i));
    subplot(nSamples,5,(i-1)*5+3);
    imshow(inputs{6}(:,:,:,i));
%     subplot(nSamples,5,(i-1)*5+4);
    subplot(nSamples,5,(i-1)*5+5);
    imshow(predictions(:,:,2,i));
end

end

% --------------------------------------------------------------------
function inputs = getDagNNBatch(imdb, batch)
% --------------------------------------------------------------------
image1 = imdb.images.image1(:,:,:,batch) ;
image2 = imdb.images.image2(:,:,:,batch) ;
image3 = imdb.images.image3(:,:,:,batch) ;
zeroImage = zeros(size(image1), 'single');

image1 = gpuArray(image1) ;
image2 = gpuArray(image2) ;
image3 = gpuArray(image3) ;
zeroImage = gpuArray(zeroImage) ;

% inputs = {'image1', image1, ...
%           'image2', image2, ...
%           'image3', image3, ...
%           'zeroImage', zeroImage};
      
inputs = {'image1', image1, ...
  'image2', image2, ...
  'image3', image3};
end