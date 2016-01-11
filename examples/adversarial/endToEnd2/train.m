function stats = train(varargin)
%TRAIN Train multi-scale recurrent models using MatConvNet

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', '..', 'matlab', 'vl_setupnn.m')) ;

% experiment and data paths
opts.baseDir = 'data';
opts.dataset = 'mnist';
% opts.dataset = 'cifar';
opts.imageSize = [32 32 1] ;
opts.seed = 2;
opts.modelType = 'test' ;
opts.learningRate = logspace(-2, -3, 100) ;
opts.batchSize = 64 ;
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
opts.train.derOutputs = {'objective', 1, 'euclidean', 1} ;

opts = vl_argparse(opts, varargin) ;
opts.train.numEpochs = numel(opts.train.learningRate) ;

% -------------------------------------------------------------------------
% Setup models
% -------------------------------------------------------------------------

% generator
opts.imageSize = [1 1 50] ;
generatorNet = getNet('type', 'generator', 'imageSize', opts.imageSize);
generatorNet.meta.normalization.imageSize = opts.imageSize ;
generatorNet.meta.normalization.averageImage = [] ;
print(generatorNet, {'input', [generatorNet.meta.normalization.imageSize 1]})

% discriminator
opts.imageSize = [16 16 1] ;
discriminatorNet = getNet('type', 'discriminator', 'imageSize', opts.imageSize);
discriminatorNet.meta.normalization.imageSize = opts.imageSize ;
discriminatorNet.meta.normalization.averageImage = [] ;
print(discriminatorNet, {'input', [discriminatorNet.meta.normalization.imageSize 1]})

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

% % Get dataset statistics
% if exist(opts.imdbStatsPath)
%   stats = load(opts.imdbStatsPath) ;
% else
%   stats = getDatasetStatistics(imdb) ;
%   save(opts.imdbStatsPath, '-struct', 'stats') ;
% end

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

% Setup data fetching options
bopts.useGpu = numel(opts.train.gpus) > 0 ;

% Launch SGD
[~, stats] = cnn_train_dag_adversarial(generatorNet, discriminatorNet, imdb, getBatchWrapper(bopts), opts.train, ...
  'train', train, ...
  'val', val) ;

end

% -------------------------------------------------------------------------
function fn = getBatchWrapper(bopts)
% -------------------------------------------------------------------------
    fn = @(imdb, batch) getBatch(imdb, batch, bopts) ;
end