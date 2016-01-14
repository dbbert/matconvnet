function [net,stats] = cnn_train_dag_adversarial(generatorNet, discriminatorNet, imdb, getBatch, varargin)
%CNN_TRAIN_DAG Demonstrates training a CNN using the DagNN wrapper
%    CNN_TRAIN_DAG() is similar to CNN_TRAIN(), but works with
%    the DagNN wrapper instead of the SimpleNN wrapper.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% Todo: save momentum with checkpointing (a waste?)

opts.expDir = fullfile('data','exp') ;
opts.continue = true ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;

opts.derOutputs = {'objective', 1} ;
opts.extractStatsFn = @extractStats ;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

generatorState.getBatch = getBatch ;
evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  if isempty(opts.derOutputs)
    error('DEROUTPUTS must be specified when training.\n') ;
  end
end
stats = [] ;

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
  if exist(opts.memoryMapFile)
    delete(opts.memoryMapFile) ;
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('resuming by loading epoch %d\n', start) ;
  [generatorNet, discriminatorNet, stats] = loadState(modelPath(start)) ;
end

for epoch=start+1:opts.numEpochs

  % train one epoch
  generatorState.epoch = epoch ;
  generatorState.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  generatorState.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  generatorState.val = opts.val ;
  generatorState.imdb = imdb ;
  
  discriminatorState = generatorState ;
%   discriminatorState.learningRate = 0 ;
  discriminatorState.learningRate = discriminatorState.learningRate;

  if numGpus <= 1
    stats.train(epoch) = process_epoch(generatorNet, discriminatorNet, generatorState, discriminatorState, opts, 'train') ;
    stats.val(epoch) = process_epoch(generatorNet, discriminatorNet, generatorState, discriminatorState, opts, 'val') ;
  else
    savedNet = net.saveobj() ;
    spmd
      net_ = dagnn.DagNN.loadobj(savedNet) ;
      stats_.train = process_epoch(net_, generatorState, opts, 'train') ;
      stats_.val = process_epoch(net_, generatorState, opts, 'val') ;
      if labindex == 1, savedNet_ = net_.saveobj() ; end
    end
    net = dagnn.DagNN.loadobj(savedNet_{1}) ;
    stats__ = accumulateStats(stats_) ;
    stats.train(epoch) = stats__.train ;
    stats.val(epoch) = stats__.val ;
    clear net_ stats_ stats__ savedNet_ ;
  end

  if ~evaluateMode
    saveState(modelPath(epoch), generatorNet, discriminatorNet, stats) ;
  end

  figure(1) ; clf ;
  plots = setdiff(...
    cat(2,...
        fieldnames(stats.train)', ...
        fieldnames(stats.val)'), {'num', 'time'}) ;
  for p = plots
    p = char(p) ;
    values = zeros(0, epoch) ;
    leg = {} ;
    for f = {'train', 'val'}
      f = char(f) ;
      if isfield(stats.(f), p)
        tmp = [stats.(f).(p)] ;
        values(end+1,:) = tmp(1,:)' ;
        leg{end+1} = f ;
      end
    end
    subplot(1,numel(plots),find(strcmp(p,plots))) ;
    plot(1:epoch, values','o-') ;
    xlabel('epoch') ;
    title(p) ;
    legend(leg{:}) ;
    grid on ;
  end
  drawnow ;
  print(1, modelFigPath, '-dpdf') ;
end

% -------------------------------------------------------------------------
function stats = process_epoch(generatorNet, discriminatorNet, generatorState, discriminatorState, opts, mode)
% -------------------------------------------------------------------------
vectorSize = 50 ;

if strcmp(mode,'train')
  generatorState.momentum = num2cell(zeros(1, numel(generatorNet.params))) ;
  discriminatorState.momentum = num2cell(zeros(1, numel(discriminatorNet.params))) ;
end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  generatorNet.move('gpu') ;
  discriminatorNet.move('gpu') ;
  if strcmp(mode,'train')
    generatorState.momentum = cellfun(@gpuArray,generatorState.momentum,'UniformOutput',false) ;
    discriminatorState.momentum = cellfun(@gpuArray,discriminatorState.momentum,'UniformOutput',false) ;
  end
end
if numGpus > 1
  mmap = map_gradients(opts.memoryMapFile, generatorNet, numGpus) ;
else
  mmap = [] ;
end

stats.time = 0 ;
stats.num = 0 ;
subset = generatorState.(mode) ;
start = tic ;
num = 0 ;

batchNumber = 1;
for t=1:opts.batchSize:numel(subset)
  
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    inputs = discriminatorState.getBatch(discriminatorState.imdb, batch) ;

    if opts.prefetch
      if s == opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
      generatorState.getBatch(generatorState.imdb, nextBatch) ;
    end
    
    % TODO: this implementation does not work correctly with subbatches yet
%     generatorNet.reset() ;
%     discriminatorNet.reset() ;
%     [generatorNet.vars.value] = deal([]) ;
%     [discriminatorNet.vars.value] = deal([]) ;
%     [generatorNet.vars.der] = deal([]) ;
%     [discriminatorNet.vars.der] = deal([]) ;
%     [generatorNet.params.der] = deal([]) ;
%     [discriminatorNet.params.der] = deal([]) ;
%     discriminatorNet.layers(end-1).block.reset() ;
%     discriminatorNet.layers(end).block.reset() ;
    if strcmp(mode, 'train')
        % alternate between updating generator and discriminator. It is
        % important to start with generator update because otherwise the
        % batch normalization params are zero and division by zero gives
        % error.
%         if mod(batchNumber,3) ~= 0 % update generator
        if mod(batchNumber,2) == 1 % update generator
%         if true
          % setup inputs and labels
%           randomVectors = gpuArray.rand(1,1,vectorSize,opts.batchSize,'single')*2-1 ;
          randomVectors = gpuArray.randn(1,1,vectorSize,opts.batchSize,'single') ;
%           randomVectors = gpuArray.ones(1,1,vectorSize,opts.batchSize,'single') ;
                    
          % forward through the generator
          generatorNet.mode = 'normal' ;
          generatorNet.accumulateParamDers = (s ~= 1) ;
          generatorNet.eval({'input', randomVectors}) ;

          % forward and backward through the discriminator
          tmpAvg = discriminatorNet.layers(end).block.average ;
          tmpNumavg = discriminatorNet.layers(end).block.numAveraged ;
          tmpAvg2 = discriminatorNet.layers(end-1).block.average ;
          tmpNumavg2 = discriminatorNet.layers(end-1).block.numAveraged ;
          
          images = generatorNet.vars(end).value ;
          labels = 1*gpuArray.ones(1,1,1,opts.batchSize,'single'); % switch labels: try to make the discriminator believe that the images it sees are from the dataset
          
%         % or throw away half of the batch and replace with real images.
%           images = cat(4, inputs{2}(:,:,:,1:end/2), generatorNet.vars(end).value) ;
%           labels = cat(4, 2*gpuArray.ones(1,1,1,opts.batchSize/2,'single'), 1*gpuArray.ones(1,1,1,opts.batchSize/2,'single')); % switch labels: try to make the discriminator believe that the images it sees are from the dataset

          discriminatorNet.mode = 'normal' ;
          discriminatorNet.accumulateParamDers = (s ~= 1) ;
          derOutputs = opts.derOutputs;
          derOutputs{4} = 0; % We don't want to optimize the generator for euclidean loss, only for generating nice looking images.
          discriminatorNet.eval({'input', images, 'embedding', randomVectors, 'label', labels}, derOutputs) ;
          
          discriminatorNet.layers(end).block.average = tmpAvg;
          discriminatorNet.layers(end).block.numAveraged = tmpNumavg;
          discriminatorNet.layers(end-1).block.average = tmpAvg2;
          discriminatorNet.layers(end-1).block.numAveraged = tmpNumavg2;

          % backward through the generator
          generatorNet.eval({'input', randomVectors}, {generatorNet.vars(end).name, discriminatorNet.vars(1).der}) ;
          
          % update the generator
          generatorState = accumulate_gradients(generatorState, generatorNet, opts, batchSize, mmap) ;
        else % update discriminator
          % setup inputs and labels
          % generate n images (n = opts.bat = half batch size)
          % forward through the generator
%           randomVectors = gpuArray.rand(1,1,vectorSize,opts.batchSize/2,'single')*2-1 ;
          randomVectors = gpuArray.randn(1,1,vectorSize,opts.batchSize/2,'single') ;
%           randomVectors = gpuArray.ones(1,1,vectorSize,opts.batchSize/2,'single') ;
          generatorNet.mode = 'test' ; % should this not be test? Probably yes, but tanh gives NaN in the beginning then.
          generatorNet.eval({'input', randomVectors}) ;
          
          % throw away half of the batch and replace it with generated images. Then shuffle again.
          images = cat(4, inputs{2}(:,:,:,1:end/2), generatorNet.vars(end).value) ;
          labels = cat(4, 1*gpuArray.ones(1,1,1,opts.batchSize/2,'single'), 2*gpuArray.ones(1,1,1,opts.batchSize/2,'single')); % 1 = from dataset, 2 = generated
          
          discriminatorNet.mode = 'normal' ;
          discriminatorNet.accumulateParamDers = (s ~= 1) ;
          embeddings = cat(4, zeros(size(randomVectors), 'single'), randomVectors) ;
          
          % set euclidean loss instanceweights of the first batch half (real images) to zero, because we can't compare their euclidean loss.
          discriminatorNet.layers(discriminatorNet.getLayerIndex('euclidean')).block.opts = {'instanceweights', single(cat(1, zeros(opts.batchSize/2, 1), ones(opts.batchSize/2, 1)))} ;
          % forward and backward through the discriminator
          discriminatorNet.eval({'input', images, 'embedding', embeddings, 'label', labels}, opts.derOutputs) ;
          discriminatorNet.layers(discriminatorNet.getLayerIndex('euclidean')).block.opts = {} ;        

          % update the discriminator
          discriminatorState = accumulate_gradients(discriminatorState, discriminatorNet, opts, batchSize, mmap) ;
          
          % c) extract learning stats
          stats = opts.extractStatsFn(discriminatorNet) ;

        %   % accumulate gradient
        %   if strcmp(mode, 'train')
        %     if ~isempty(mmap)
        %       write_gradients(mmap, generatorNet) ;
        %       labBarrier() ;
        %     end
        %     generatorState = accumulate_gradients(generatorState, generatorNet, opts, batchSize, mmap) ;
        %   end

          % print learning statistics
          time = toc(start) ;
          stats.num = num ;
          stats.time = toc(start) ;

          fprintf('%s: epoch %02d: %3d/%3d: %.1f Hz', ...
            mode, ...
            generatorState.epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize), ...
            stats.num/stats.time * max(numGpus, 1)) ;
          for f = setdiff(fieldnames(stats)', {'num', 'time'})
            f = char(f) ;
            fprintf(' %s:', f) ;
            fprintf(' %.3f', stats.(f)) ;
          end
          fprintf('\n') ;
        end         
    else
      generatorNet.mode = 'test' ;
      generatorNet.eval(inputs) ;
      
      discriminatorNet.mode = 'test' ;
      discriminatorNet.eval(inputs) ;
    end
  end
  
  batchNumber = batchNumber + 1;
  
end
% visualize(generatorNet) ;
generatorNet.reset() ;
generatorNet.move('cpu') ;
discriminatorNet.reset() ;
discriminatorNet.move('cpu') ;

% -------------------------------------------------------------------------
function state = accumulate_gradients(state, net, opts, batchSize, mmap)
% -------------------------------------------------------------------------
for p=1:numel(net.params)

  % bring in gradients from other GPUs if any
  if ~isempty(mmap)
    numGpus = numel(mmap.Data) ;
    tmp = zeros(size(mmap.Data(labindex).(net.params(p).name)), 'single') ;
    for g = setdiff(1:numGpus, labindex)
      tmp = tmp + mmap.Data(g).(net.params(p).name) ;
    end
    net.params(p).der = net.params(p).der + tmp ;
  else
    numGpus = 1 ;
  end

  switch net.params(p).trainMethod

    case 'average' % mainly for batch normalization
      thisLR = net.params(p).learningRate ;
      net.params(p).value = ...
          (1 - thisLR) * net.params(p).value + ...
          (thisLR/batchSize) * net.params(p).der ;

    case 'gradient'
      thisDecay = opts.weightDecay * net.params(p).weightDecay ;
      thisLR = state.learningRate * net.params(p).learningRate ;
      state.momentum{p} = opts.momentum * state.momentum{p} ...
        - thisDecay * net.params(p).value ...
        - (1 / batchSize) * net.params(p).der ;
      net.params(p).value = net.params(p).value + thisLR * state.momentum{p} ;

    case 'otherwise'
      error('Unknown training method ''%s'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.params)
  format(end+1,1:3) = {'single', size(net.params(i).value), net.params(i).name} ;
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net)
% -------------------------------------------------------------------------
for i=1:numel(net.params)
  mmap.Data(labindex).(net.params(i).name) = gather(net.params(i).der) ;
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

stats = struct() ;

for s = {'train', 'val'}
  s = char(s) ;
  total = 0 ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;

      if g == 1
        stats.(s).(f) = 0 ;
      end
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
stats = struct() ;
for i = 1:numel(sel)
  stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function saveState(fileName, generatorNet, discriminatorNet, stats)
% -------------------------------------------------------------------------
generatorNet_ = generatorNet ;
generatorNet = generatorNet_.saveobj() ;

discriminatorNet_ = discriminatorNet ;
discriminatorNet = discriminatorNet_.saveobj() ;


save(fileName, 'generatorNet', 'discriminatorNet', 'stats') ;

% -------------------------------------------------------------------------
function [generatorNet, discriminatorNet, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'generatorNet', 'discriminatorNet', 'stats') ;
generatorNet = dagnn.DagNN.loadobj(generatorNet) ;
discriminatorNet = dagnn.DagNN.loadobj(discriminatorNet) ;

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
