function [net,stats] = cnn_train_dag_autoencoder(encoderNet, decoderNet, imdb, getBatch, varargin)
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

encoderState.getBatch = getBatch ;
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
  [encoderNet, decoderNet, stats] = loadState(modelPath(start)) ;
end

for epoch=start+1:opts.numEpochs

  % train one epoch
  encoderState.epoch = epoch ;
  encoderState.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  encoderState.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  encoderState.val = opts.val ;
  encoderState.imdb = imdb ;
  
  decoderState = encoderState ;

  if numGpus <= 1
    stats.train(epoch) = process_epoch(encoderNet, decoderNet, encoderState, decoderState, opts, 'train', imdb) ;
    stats.val(epoch) = process_epoch(encoderNet, decoderNet, encoderState, decoderState, opts, 'val', imdb) ;
  else
    savedNet = net.saveobj() ;
    spmd
      net_ = dagnn.DagNN.loadobj(savedNet) ;
      stats_.train = process_epoch(net_, encoderState, opts, 'train') ;
      stats_.val = process_epoch(net_, encoderState, opts, 'val') ;
      if labindex == 1, savedNet_ = net_.saveobj() ; end
    end
    net = dagnn.DagNN.loadobj(savedNet_{1}) ;
    stats__ = accumulateStats(stats_) ;
    stats.train(epoch) = stats__.train ;
    stats.val(epoch) = stats__.val ;
    clear net_ stats_ stats__ savedNet_ ;
  end

  if ~evaluateMode
    saveState(modelPath(epoch), encoderNet, decoderNet, stats) ;
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
function stats = process_epoch(encoderNet, decoderNet, encoderState, decoderState, opts, mode, imdb)
% -------------------------------------------------------------------------

if strcmp(mode,'train')
  encoderState.momentum = num2cell(zeros(1, numel(encoderNet.params))) ;
  decoderState.momentum = num2cell(zeros(1, numel(decoderNet.params))) ;
end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  encoderNet.move('gpu') ;
  decoderNet.move('gpu') ;
  if strcmp(mode,'train')
    encoderState.momentum = cellfun(@gpuArray,encoderState.momentum,'UniformOutput',false) ;
    decoderState.momentum = cellfun(@gpuArray,decoderState.momentum,'UniformOutput',false) ;
  end
end
if numGpus > 1
  mmap = map_gradients(opts.memoryMapFile, encoderNet, numGpus) ;
else
  mmap = [] ;
end

stats.time = 0 ;
stats.num = 0 ;
subset = encoderState.(mode) ;
start = tic ;
num = 0 ;

for t=1:opts.batchSize:numel(subset)
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;

  % FIRST do a regular autoencoder update with images
    % get this image batch and prefetch the next
%     decoderNet.addLayer('loss', dagnn.Loss('loss', 'L1'), [decoderNet.vars(end).name, {'image'}], 'objective') ;
    
    batchStart = t + (labindex-1) ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    inputs = encoderState.getBatch(encoderState.imdb, batch) ;

    if strcmp(mode, 'train')
      encoderNet.mode = 'normal' ;
      encoderNet.accumulateParamDers = 0 ;
      decoderNet.mode = 'normal' ;
      decoderNet.accumulateParamDers = 0 ;
      
      % forward through the encoder
      encoderNet.eval({'image', inputs{2}}, []) ;
      encodings = encoderNet.vars(end).value ;
      % forward and backward through the decoder
      decoderNet.eval({'encoding', encodings, 'image', inputs{2}}, {'objective', 1}) ;
      derivatives = decoderNet.vars(1).der ;
      % backward through the encoder
      encoderNet.eval({'image', inputs{2}}, {encoderNet.vars(end).name, derivatives}) ;
      
      % extract learning stats
      stats = opts.extractStatsFn(decoderNet) ;
    else
      % TODO
      encoderNet.mode = 'test' ;
      encoderNet.eval(inputs) ;
    end
    
%     decoderNet.removeLayer('loss') ;
%     encoderNet.addLayer('loss', dagnn.Loss('loss', 'L1'), [encoderNet.vars(end).name, {'encoding'}], 'objective') ;
%   % SECOND do an inverse autoencoder update with white noise (and enable derivative accumulation)
%     if strcmp(mode, 'train')
%       encoderNet.accumulateParamDers = 1 ; % accumulate the ders with the previous update.
%       decoderNet.accumulateParamDers = 1 ;
%       
%       randomEncodings = gpuArray.randn(size(encodings),'single');
%       
%       % forward through the decoder
%       decoderNet.eval({'encoding', randomEncodings}, []) ;
%       decodings = decoderNet.vars(end).value ;
%       % forward and backward through the encoder
%       encoderNet.eval({'image', decodings, 'encoding', randomEncodings}, {'objective', 1}) ;
%       derivatives = encoderNet.vars(1).der ;
%       % backward through the decoder
%       decoderNet.eval({'encoding', randomEncodings}, {decoderNet.vars(end).name, derivatives}) ; 
%       
%       % extract learning stats
%       statsTmp = opts.extractStatsFn(encoderNet) ;
%       stats.objective2 = statsTmp.objective ;
%     else
%       % TODO
%       encoderNet.mode = 'test' ;
%       encoderNet.eval(inputs) ;
%     end
%     
%     encoderNet.removeLayer('loss') ;

  % accumulate gradient
  if strcmp(mode, 'train')
    if ~isempty(mmap)
      write_gradients(mmap, net) ;
      labBarrier() ;
    end
    encoderState = accumulate_gradients(encoderState, encoderNet, opts, batchSize, mmap) ;
    decoderState = accumulate_gradients(decoderState, decoderNet, opts, batchSize, mmap) ;
%     % batchSize * 2 because we have essentially doubled it.
%     encoderState = accumulate_gradients(encoderState, encoderNet, opts, batchSize*2, mmap) ;
%     decoderState = accumulate_gradients(decoderState, decoderNet, opts, batchSize*2, mmap) ;
  end

  % print learning statistics
  time = toc(start) ;
  stats.num = num ;
  stats.time = toc(start) ;

  fprintf('%s: epoch %02d: %3d/%3d: %.1f Hz', ...
    mode, ...
    encoderState.epoch, ...
    fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize), ...
    stats.num/stats.time * max(numGpus, 1)) ;
  for f = setdiff(fieldnames(stats)', {'num', 'time'})
    f = char(f) ;
    fprintf(' %s:', f) ;
    fprintf(' %.3f', stats.(f)) ;
  end
  fprintf('\n') ;
end

% visualize(encoderNet, decoderNet, imdb) ;
encoderNet.reset() ;
encoderNet.move('cpu') ;
decoderNet.reset() ;
decoderNet.move('cpu') ;

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
function saveState(fileName, encoderNet, decoderNet, stats)
% -------------------------------------------------------------------------
encoderNet_ = encoderNet ;
encoderNet = encoderNet_.saveobj() ;

decoderNet_ = decoderNet ;
decoderNet = decoderNet_.saveobj() ;

save(fileName, 'encoderNet', 'decoderNet', 'stats') ;

% -------------------------------------------------------------------------
function [encoderNet, decoderNet, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'encoderNet', 'decoderNet', 'stats') ;
encoderNet = dagnn.DagNN.loadobj(encoderNet) ;
decoderNet = dagnn.DagNN.loadobj(decoderNet) ;

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
