function [net, info] = trainDeepTracking(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

% experiment and data paths
opts.baseDir = 'data';
opts.dataset = 'mnistMoving2';
% opts.dataset = 'cifar';
opts.imageSize = [64 64 1] ;
opts.seed = 2;
opts.modelType = 'test' ;
opts.learningRate = logspace(-4, -6, 20) ;
opts.batchSize = 16 ; % 512
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
opts.train.weightDecay = 1e-4; % 1e-4
opts.train.momentum = 0.50;
opts.train.derOutputs = {'objective', 1} ;
% opts.train.derOutputs = {'objective', 1, 'mmd', 1000000} ;
% opts.train.derOutputs = {'objective', 1, 'mmd', 10000} ;
% opts.train.derOutputs = {'objective', 1, 'repela', 0} ;
% opts.train.derOutputs = {'objective', 1, 'repel', 1} ;
% opts.train.derOutputs = {'objective', 1, 'repelb', 1, 'repel', 1} ;
% opts.train.derOutputs = {'objective', 1, 'repela', 1, 'repelb', 1, 'repel', 1} ;

opts = vl_argparse(opts, varargin) ;
opts.train.numEpochs = numel(opts.train.learningRate) ;

% -------------------------------------------------------------------------
% Setup models
% -------------------------------------------------------------------------
% defaultNet = getNet('type', 'deepTracking', 'imageSize', opts.imageSize);
% defaultNet.meta.normalization.imageSize = opts.imageSize ;
% defaultNet.meta.normalization.averageImage = [] ;
% print(defaultNet, {'image1', [defaultNet.meta.normalization.imageSize 1], ...
%                    'image2', [defaultNet.meta.normalization.imageSize 1], ...
%                    'image3', [defaultNet.meta.normalization.imageSize 1], ...
%                    'image4', [defaultNet.meta.normalization.imageSize 1], ...
%                    'zeroImage', [defaultNet.meta.normalization.imageSize 1]})
               
defaultNet = getNet('type', 'deepTrackingConcat', 'imageSize', opts.imageSize);
defaultNet.meta.normalization.imageSize = opts.imageSize ;
defaultNet.meta.normalization.averageImage = [] ;
print(defaultNet, {'image1', [defaultNet.meta.normalization.imageSize 1], ...
                   'image2', [defaultNet.meta.normalization.imageSize 1], ...
                   'image3', [defaultNet.meta.normalization.imageSize 1], ...
                   'image4', [defaultNet.meta.normalization.imageSize 1]})

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

% imdb.images.image = randn(1,1,50,size(imdb.images.image,4), 'single');

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
% [~, stats] = cnn_train_dag_autoencoder(encoderNet, decoderNet, imdb, getBatchWrapper(bopts), opts.train, ...
%   'train', train, ...
%   'val', val) ;

[net, stats] = cnn_train_dag(defaultNet, imdb, getBatchWrapper(bopts), opts.train, ...
  'train', train, ...
  'val', val) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(bopts)
% -------------------------------------------------------------------------
    fn = @(imdb, batch) getDagNNBatch(imdb, batch, bopts) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(imdb, batch, opts)
% --------------------------------------------------------------------
image1 = imdb.images.image1(:,:,:,batch) ;
image2 = imdb.images.image2(:,:,:,batch) ;
image3 = imdb.images.image3(:,:,:,batch) ;
image4 = imdb.images.image4(:,:,:,batch) ;
zeroImage = zeros(size(image1), 'single');

if opts.useGpu == 1
  image1 = gpuArray(image1) ;
  image2 = gpuArray(image2) ;
  image3 = gpuArray(image3) ;
  image4 = gpuArray(image4) ;
  zeroImage = gpuArray(zeroImage) ;
end

% inputs = {'image1', image1, ...
%           'image2', image2, ...
%           'image3', image3, ...
%           'image4', image4, ...
%           'zeroImage', zeroImage};
      
inputs = {'image1', image1, ...
  'image2', image2, ...
  'image3', image3, ...
  'image4', image4};