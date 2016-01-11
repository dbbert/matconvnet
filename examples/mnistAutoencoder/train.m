function [net, info] = train(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

% experiment and data paths
opts.baseDir = 'data';
opts.dataset = 'mnist';
% opts.dataset = 'cifar';
opts.imageSize = [16 16 1] ;
opts.seed = 1;
opts.modelType = 'test' ;
opts.learningRate = logspace(-3, -4, 10) ;
opts.batchSize = 32 ;
opts.nClasses = 8 ;

[opts, varargin] = vl_argparse(opts, varargin) ;

opts.baseName = fullfile(opts.baseDir, sprintf('%s-%d', opts.dataset, opts.seed)) ;
opts.dataDir = sprintf('%s-%s', opts.baseName, 'data') ;
opts.expDir = sprintf('%s-%s', opts.baseName, opts.modelType) ;

% experiment setup
opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat') ;
opts.imdbStatsPath = fullfile(opts.dataDir, 'imdbStats.mat') ;

% training options (SGD)
opts.train.batchSize = opts.batchSize ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.gpus = [1] ;
opts.train.prefetch = false ;
opts.train.expDir = opts.expDir ;
opts.train.learningRate = opts.learningRate ;
opts.train.weightDecay = 1e-5;
opts.train.momentum = 0.50;
opts.train.derOutputs = {'objective', 1} ;

opts = vl_argparse(opts, varargin) ;
opts.train.numEpochs = numel(opts.train.learningRate) ;

% -------------------------------------------------------------------------
% Setup models
% -------------------------------------------------------------------------

%
net = getNet('imageSize', opts.imageSize);
net.meta.normalization.imageSize = opts.imageSize ;
net.meta.normalization.averageImage = [] ;
print(net, {'input', [net.meta.normalization.imageSize 1]})

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% Get dataset
if exist(opts.imdbPath)
    imdb = load(opts.imdbPath) ;
else
    imdb = feval(sprintf('setupImdb_%s', opts.dataset), struct('dataDir', opts.dataDir));
    mkdir(opts.dataDir) ;
    save(opts.imdbPath, '-struct', 'imdb', '-v7.3') ;
end

% Get training and test/validation subsets
train = find(imdb.images.set == 1) ;
% train = train(1:1024*10) ;
val = find(imdb.images.set == 2) ;

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

% Setup data fetching options
bopts.useGpu = numel(opts.train.gpus) > 0 ;

% Launch SGD
[net, info] = cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
  'train', train, ...
  'val', val) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(bopts)
% -------------------------------------------------------------------------
    fn = @(imdb, batch) getDagNNBatch(imdb, batch, bopts) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(imdb, batch, opts)
% --------------------------------------------------------------------
images = imdb.images.image(:,:,:,batch) ;
labels = imdb.images.label(1,batch) ;
if opts.useGpu == 1
  images = gpuArray(images) ;
end
% inputs = {'input', images, 'label', labels} ;
inputs = {'input', images} ;