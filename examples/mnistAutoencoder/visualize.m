function visualize(net)

if nargin == 0
    clear;
    opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnist-1-test/net-epoch-3.mat';
    nets = load(opts.netFile);
    net = dagnn.DagNN.loadobj(nets.net) ;
end
net.move('gpu') ;
net.mode = 'test';

imdb = load('/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnist-0-data/imdb.mat');

%% display filters
filters = net.params(1).value;
tiledImage = tileImage(gather(filters), 10, 2);
figure;
subplot(2,1,1);
imshow(tiledImage);

%% autoencode some samples
sample = randsample(60000, 16);
inputs = imdb.images.image(:,:,:,sample);

net.vars(end-1).precious = true;
net.eval({'input', gpuArray(inputs)});
reconstructions = net.vars(end-1).value;

results = cat(4, inputs, reconstructions) ;

tiledImage = tileImage(gather(results),16,2);
figure;
imshow(tiledImage);