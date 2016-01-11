clear;

imdb = load('/users/visics/bdebraba/devel/matconvnet/examples/adversarial/dataNoise/mnist-3-data/imdb.mat');

opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/adversarial/data/mnist-6-test/net-epoch-60.mat';
% opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/adversarial/data/toyotaThumbs-1-test/net-epoch-11.mat';

vectorSize = 50;

nets = load(opts.netFile);
generatorNet = dagnn.DagNN.loadobj(nets.generatorNet) ;
generatorNet.move('gpu') ;
generatorNet.mode = 'test';

opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/adversarial/data/generated-1-test/net-epoch-5.mat' ;
% opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/adversarial/data/generated-2-test/net-epoch-5.mat' ;
net = load(opts.netFile);
encoderNet = dagnn.DagNN.loadobj(net.net) ;
encoderNet.move('gpu') ;
encoderNet.mode = 'test';

%% display random generations
n = 16;

sample = randsample(60000, n);
inputs = imdb.images.image(:,:,:,sample);

encoderNet.eval({'input', gpuArray(inputs)});
encodings = encoderNet.vars(end-2).value;

generatorNet.eval({'input', encodings});
reconstructions = generatorNet.vars(end).value;

results = cat(4, inputs, reconstructions) ;

tiledImage = tileImage(gather(results),16,2);
imshow(tiledImage);