% -------------------------------------------------------------------------
function net = getNetNoise(varargin)
% -------------------------------------------------------------------------
    opts.type = 'generator'; % or discriminator
    opts.imageSize = [32 32 1];
    
    opts = vl_argparse(opts, varargin) ;

    netOpts.scale = 0.1 ;
    netOpts.cudnnWorkspaceLimit = 3*1024*1024*1024 ; % 3GB
    netOpts.initBias = 0.1 ;
    netOpts.weightDecay = 1 ;
    netOpts.batchNormalization = true;
    netOpts.weightInitMethod = 'xavierimproved';
    netOpts.nChns = 1;
    netOpts.nFilterMultiplier = 1;
    
    vectorSize = 10;
    
    net = dagnn.DagNN() ;
    
    inputs = {'input'};
    
%     switch opts.type
%         case 'discriminator'
%             outputs = add_block(net, netOpts, 'block1', inputs, 5, 5, netOpts.nChns, 8, 2, 2);
% %             outputs = add_conv(net, netOpts, 'block1', inputs, 5, 5, netOpts.nChns, 8, 2, 2);
% %             outputs = add_relu(net, netOpts, 'block1', outputs, 0.2);
%     
%             outputs = add_block(net, netOpts, 'block2', outputs, 5, 5, 8, 16, 2, 2);
%             outputs = add_block(net, netOpts, 'block3', outputs, 5, 5, 16, 32, 2, 2);
% %             outputs = add_block(net, netOpts, 'block4', outputs, 4, 4, 128, 2, 1, 0);
%             outputs = add_conv(net, netOpts, 'block4', outputs, 4, 4, 32, 2, 1, 0);
%             net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), [outputs, {'label'}], 'objective') ;
%             net.addLayer('error', dagnn.Loss('loss', 'classerror'), [outputs, {'label'}], 'error') ;
%         case 'generator'
% %             outputs = add_norm(net, netOpts, 'deblockNorm', inputs, );
%             outputs = add_normalize(net, netOpts, 'normalize', inputs);
%             outputs = add_deblock(net, netOpts, 'deblock4', outputs, 4, 4, 50, 128, 1, 0);
%             outputs = add_deblock(net, netOpts, 'deblock3', outputs, 4, 4, 128, 64, 2, 1);
%             outputs = add_deblock(net, netOpts, 'deblock2', outputs, 4, 4, 64, 32, 2, 1);
%             outputs = add_deblockSigmoid(net, netOpts, 'deblock1', outputs, 4, 4, 32, netOpts.nChns, 2, 1);
%     end
    
        switch opts.type
            case 'discriminator'
                outputs = add_conv(net, netOpts, 'block1', inputs, 1, 1, vectorSize, 64, 1, 0); % this is a simple linear classifier.
                outputs = add_relu(net, netOpts, 'block1', outputs, 0.2);
                outputs = add_conv(net, netOpts, 'block2', outputs, 1, 1, 64, 2, 1, 0);
%                 outputs = add_relu(net, netOpts, 'block2', outputs, 0.2);
%                 outputs = add_conv(net, netOpts, 'block3', outputs, 1, 1, 128, 64, 1, 0);
%                 outputs = add_relu(net, netOpts, 'block3', outputs, 0.2);
%                 outputs = add_conv(net, netOpts, 'block4', outputs, 1, 1, 64, 2, 1, 0);
                
                net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), [outputs, {'label'}], 'objective') ;
                net.addLayer('error', dagnn.Loss('loss', 'classerror'), [outputs, {'label'}], 'error') ;
            case 'generator'
                outputs = add_block(net, netOpts, 'block0', inputs, 32, 32, netOpts.nChns, vectorSize, 1, 0);
%                 outputs = add_block(net, netOpts, 'block1', inputs, 3, 3, netOpts.nChns, 16, 2, 1);
%                 outputs = add_block(net, netOpts, 'block2', outputs, 3, 3, 16, 32, 2, 1);
%                 outputs = add_block(net, netOpts, 'block3', outputs, 3, 3, 32, 64, 2, 1);
% %                 outputs = add_block(net, netOpts, 'block3', outputs, 8, 8, 32, 10, 1, 0);
%                 outputs = add_dropout(net, netOpts, 'block4', outputs, 0.5);
%                 outputs = add_conv(net, netOpts, 'block4', outputs, 4, 4, 64, vectorSize, 1, 0);
% %                 outputs = add_block(net, netOpts, 'block3', outputs, 3, 3, 32, 64, 2, 1);
% %                 outputs = add_block(net, netOpts, 'block4', outputs, 4, 4, 64, 128, 1, 0); % fully connected
% %                 outputs = add_block(net, netOpts, 'block5', outputs, 1, 1, 128, 10, 1, 0); % fully connected
                
                outputs = add_normalize(net, netOpts, 'normalize', outputs);
        end
    
%     net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog', 'opts', {'instanceWeights', 1/prod(opts.imageSize(1:2))}), ...
%              [outputHigh, {'label'}], 'objective') ;
%     net.addLayer('accuracy', SegmentationAccuracy('nClasses', opts.nClasses), ...
%              [outputHigh, {'label'}], 'accuracy') ;
%     net.addLayer('error', dagnn.Loss('loss', 'classerror', 'opts', {'instanceWeights', 1/prod(opts.imageSize(1:2))}), ...
%              [outputHigh, {'label'}], 'error') ;
end

function outputs = add_upsample(net, opts, name, inputs, in, learnable)
    opts.weightInitMethod = 'bilinear';

%     filters = single(bilinear_u(64, 21, 21)) ;
    h = 2; w = 2;
    filters = init_weight(opts, h, w, in, in, 'single') ;
    
    varName = name;
    if strcmp(name(end-3), '_') % if the name ends with suffix _*, weights should be tied
        varName = name(1:end-4);
    end
    
    layerName = sprintf('%s_upsample', name);
    outputs = {layerName};
    
    net.addLayer(layerName, ...
      dagnn.ConvTranspose(...
      'size', size(filters), ...
      'upsample', 2, ...
      'crop', [0 0 0 0], ...
      'numGroups', in, ...
      'hasBias', false), ...
      inputs, outputs, sprintf('%sf',varName)) ;    

    f = net.getParamIndex(sprintf('%sf',varName)) ;
    net.params(f).value = filters ;
    if learnable
        net.params(f).learningRate = 1 ;
    else
        net.params(f).learningRate = 0 ;
    end
    net.params(f).weightDecay = 1 ;
    
    outputsBn = add_batchNorm(net, opts, name, outputs, in);
    outputsRelu = add_relu(net, opts, name, outputsBn);
    outputs = outputsRelu;
end

function outputs = add_downsample(net, opts, name, inputs, in, learnable)    
    h = 3; w = 3;
    opts.weightInitMethod = 'bilinear';
    filters = init_weight(opts, h, w, in, in, 'single') ;
    
    varName = name;
    if strcmp(name(end-3), '_') % if the name ends with suffix _*, weights should be tied
        varName = name(1:end-4);
    end
    
    layerName = sprintf('%s_downsample', name);
    outputs = {layerName};
    
    net.addLayer(layerName, ...
      dagnn.Conv(...
      'size', size(filters), ...
      'stride', 2, ...
      'pad', 1, ...
      'hasBias', false), ...
      inputs, outputs, sprintf('%sf',varName)) ;    

    f = net.getParamIndex(sprintf('%sf',varName)) ;
    net.params(f).value = filters ;
    if learnable
        net.params(f).learningRate = 1 ;
    else
        net.params(f).learningRate = 0 ;
    end
    net.params(f).weightDecay = 1 ;
end

% block with two conv-bn-relu layers
function outputs = add_block2(net, netOpts, names, inputs, h, w, in, out, stride, pad, dropout)
    assert(numel(names) == 2);
    outputs = add_block(net, netOpts, names{1}, inputs, h, w, in, out, 1, pad);
    if dropout > 0
        outputs = add_dropout(net, netOpts, names{1}, outputs, dropout);
    end
    outputs = add_block(net, netOpts, names{2}, outputs, h, w, out, out, stride, pad);
end

function outputs = add_deblock2(net, netOpts, names, inputs, h, w, in, out, stride, pad, dropout)
    assert(numel(names) == 2);
    outputs = add_deblock(net, netOpts, names{1}, inputs, h, w, in, in, stride, pad);
    if dropout > 0
        outputs = add_dropout(net, netOpts, names{1}, outputs, dropout);
    end
    outputs = add_deblock(net, netOpts, names{2}, outputs, h, w, in, out, 1, pad);
end

% block with one conv-bn-relu layer
function outputs = add_block(net, opts, name, inputs, h, w, in, out, stride, pad)
%     outputsDropout = add_dropout(net, opts, name, inputs, 0.5);
    outputsConv = add_conv(net, opts, name, inputs, h, w, in, out, stride, pad);
    outputsBn = add_batchNorm(net, opts, name, outputsConv, out);
    outputsRelu = add_relu(net, opts, name, outputsBn, 0.2);
    
    outputs = outputsRelu;
end

function outputs = add_deblock(net, opts, name, inputs, h, w, in, out, stride, pad)
    outputsDeconv = add_deconv(net, opts, name, inputs, h, w, in, out, stride, pad);
    outputsBn = add_batchNorm(net, opts, name, outputsDeconv, out);
    outputsRelu = add_relu(net, opts, name, outputsBn);
    
    outputs = outputsRelu;
end

function outputs = add_deblockSigmoid(net, opts, name, inputs, h, w, in, out, stride, pad)
    outputsDeconv = add_deconv(net, opts, name, inputs, h, w, in, out, stride, pad);
%     outputsRelu = add_relu(net, opts, name, outputsDeconv);
%     outputsBn = add_batchNorm(net, opts, name, outputsDeconv, out);
    outputsTanh = add_tanh(net, opts, name, outputsDeconv);
    
    outputs = outputsTanh;
end

function outputs = add_conv(net, opts, name, inputs, h, w, in, out, stride, pad)
    convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;    
    params = struct(...
            'name', {}, ...
            'value', {}, ...
            'learningRate', [], ...
            'weightDecay', [], ...
            'opts', {convOpts}) ;
        
    varName = name;
    if strcmp(name(end-3), '_') % if the name ends with suffix _*, weights should be tied
        varName = name(1:end-4);
    end

    params(1).name = sprintf('%sf',varName) ;
    params(1).value = init_weight(opts, h, w, in, out, 'single') ;
    params(2).name = sprintf('%sb',varName) ;
    params(2).value = zeros(out, 1, 'single') ;

    learningRate = [1 2] ;
    weightDecay = [opts.weightDecay 0] ;

    params(1).learningRate = learningRate(1) ;
    params(2).learningRate = learningRate(2) ;
    params(1).weightDecay = weightDecay(1) ;
    params(2).weightDecay = weightDecay(2) ;

    block = dagnn.Conv() ;
    block.size = size(params(1).value) ;
    block.pad = pad ;
    block.stride = stride ;

    layerName = sprintf('%s_conv', name);
    outputs = {layerName};

    net.addLayer(...
        layerName, ...
        block, ...
        inputs, ...
        outputs, ...
        {params.name}) ;

    for p = 1:numel(params)
        pindex = net.getParamIndex(params(p).name) ;
        if ~isempty(params(p).value)
            net.params(pindex).value = params(p).value ;
        end
        if ~isempty(params(p).learningRate)
            net.params(pindex).learningRate = params(p).learningRate ;
        end
        if ~isempty(params(p).weightDecay)
            net.params(pindex).weightDecay = params(p).weightDecay ;
        end
    end
end

function outputs = add_deconv(net, opts, name, inputs, h, w, in, out, upsample, crop)
%     opts.weightInitMethod = 'bilinear';
    convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
    params = struct(...
            'name', {}, ...
            'value', {}, ...
            'learningRate', [], ...
            'weightDecay', [], ...
            'opts', {convOpts}) ;
        
    varName = name;
    if strcmp(name(end-3), '_') % if the name ends with suffix _*, weights should be tied
        varName = name(1:end-4);
    end

    params(1).name = sprintf('%sf',varName) ;
    params(1).value = init_weight(opts, h, w, out, in, 'single') ;
    params(2).name = sprintf('%sb',varName) ;
    params(2).value = zeros(1, out, 'single') ;

    learningRate = [1 2] ;
    weightDecay = [opts.weightDecay 0] ;

    params(1).learningRate = learningRate(1) ;
    params(2).learningRate = learningRate(2) ;
    params(1).weightDecay = weightDecay(1) ;
    params(2).weightDecay = weightDecay(2) ;

    block = dagnn.ConvTranspose() ;
    block.size = size(params(1).value) ;
    if numel(block.size) == 3
        block.size = [block.size 1];
    end
    block.upsample = upsample ;
    block.crop = crop ;
    % block.numgroups = numgroups ; TODO future

    layerName = sprintf('%s_deconv', name);
    outputs = {layerName};

    net.addLayer(...
        layerName, ...
        block, ...
        inputs, ...
        outputs, ...
        {params.name}) ;

    for p = 1:numel(params)
        pindex = net.getParamIndex(params(p).name) ;
        if ~isempty(params(p).value)
            net.params(pindex).value = params(p).value ;
        end
        if ~isempty(params(p).learningRate)
            net.params(pindex).learningRate = params(p).learningRate ;
        end
        if ~isempty(params(p).weightDecay)
            net.params(pindex).weightDecay = params(p).weightDecay ;
        end
    end
end

function outputs = add_relu(net, opts, name, inputs, leak)
    lopts = {} ;
    % if isfield(net.layers{l}, 'leak'), lopts = {'leak', net.layers{l}} ; end
    if nargin < 5
        leak = 0.01 ;
    end
    lopts = {'leak', leak} ;
    block = dagnn.ReLU('opts', lopts) ;

    layerName = sprintf('%s_relu', name);
    outputs = {layerName};

    net.addLayer(...
        layerName, ...
        block, ...
        inputs, ...
        outputs) ;
end

function outputs = add_sigmoid(net, opts, name, inputs)
    block = dagnn.Sigmoid() ;

    layerName = sprintf('%s_sigmoid', name);
    outputs = {layerName};

    net.addLayer(...
        layerName, ...
        block, ...
        inputs, ...
        outputs) ;
end

function outputs = add_tanh(net, opts, name, inputs)
    block = dagnn.Tanh() ;

    layerName = sprintf('%s_tanh', name);
    outputs = {layerName};

    net.addLayer(...
        layerName, ...
        block, ...
        inputs, ...
        outputs) ;
end

function outputs = add_batchNorm(net, opts, name, inputs, out)
    params = struct(...
        'name', {}, ...
        'value', {}, ...
        'learningRate', [], ...
        'weightDecay', []) ;
    
    varName = name;
    if strcmp(name(end-3), '_') % if the name ends with suffix _*, weights should be tied
        varName = name(1:end-4);
    end

    params(1).name = sprintf('%sm',varName) ;
    params(1).value = ones(out, 1, 'single') ;
    params(2).name = sprintf('%sb',varName) ;
    params(2).value = zeros(out, 1, 'single') ;
    params(3).name = sprintf('%sx',varName) ;
    params(3).value = zeros(out, 2, 'single') ;

    learningRate = [2 1 0.05] ;
    weightDecay = [0 0 0] ;

    params(1).learningRate = learningRate(1) ;
    params(2).learningRate = learningRate(2) ;
    params(3).learningRate = learningRate(3) ;
    params(1).weightDecay = weightDecay(1) ;
    params(2).weightDecay = weightDecay(2) ;
    params(3).weightDecay = weightDecay(3) ;

    block = dagnn.BatchNorm() ;

    layerName = sprintf('%s_bn', name);
    outputs = {layerName};

    net.addLayer(...
        layerName, ...
        block, ...
        inputs, ...
        outputs, ...
        {params.name}) ;
    
    for p = 1:numel(params)
        pindex = net.getParamIndex(params(p).name) ;
        if ~isempty(params(p).value)
            net.params(pindex).value = params(p).value ;
        end
        if ~isempty(params(p).learningRate)
            net.params(pindex).learningRate = params(p).learningRate ;
        end
        if ~isempty(params(p).weightDecay)
            net.params(pindex).weightDecay = params(p).weightDecay ;
        end
    end
end

function outputs = add_pool(net, opts, name, inputs, h, w, stride, pad)
    block = dagnn.Pooling() ;
    block.method = 'max' ;
    block.poolSize = [h w] ;
    block.pad = pad ;
    block.stride = stride ;

    layerName = sprintf('%s_pool', name);
    outputs = {layerName};

    net.addLayer(...
        layerName, ...
        block, ...
        inputs, ...
        outputs) ;
end

function outputs = add_concat(net, opts, name, inputs)
    block = dagnn.Concat();
    
    layerName = sprintf('%s_concat', name);
    outputs = {layerName};

    net.addLayer(...
        layerName, ...
        block, ...
        inputs, ...
        outputs) ;
end

function outputs = add_dropout(net, opts, name, inputs, rate)
    layerName = sprintf('%s_dropout', name);
    outputs = {layerName};

    frozen = false;
    
    block = dagnn.DropOut() ;
    block.rate = rate ;
    block.frozen = frozen ;
      
    net.addLayer(...
    layerName, ...
    block, ...
    inputs, ...
    outputs) ;
end

function outputs = add_normalize(net, opts, name, inputs)
    param = [1000 eps 1 0.5] ;
    block = dagnn.LRN('param', param) ;

    layerName = sprintf('%s_norm', name);
    outputs = {layerName};

    net.addLayer(...
        layerName, ...
        block, ...
        inputs, ...
        outputs) ;
end

function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

    switch lower(opts.weightInitMethod)
      case 'gaussian'
        sc = 0.01/opts.scale ;
        weights = randn(h, w, in, out, type)*sc;
      case 'xavier'
        sc = sqrt(3/(h*w*in)) ;
        weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
      case 'xavierimproved'
        sc = sqrt(2/(h*w*out)) ;
        weights = randn(h, w, in, out, type)*sc ;
      case 'bilinear'
        assert(h == w);
        assert(in == out);
        k = h;
        numGroups = in;
        numClasses = in;
        factor = floor((k+1)/2) ;
        if rem(k,2)==1
          center = factor ;
        else
          center = factor + 0.5 ;
        end
        C = 1:k ;
        if numGroups ~= numClasses
          weights = zeros(k,k,numGroups,numClasses, type) ;
        else
          weights = zeros(k,k,1,numClasses, type) ;
        end

        for i =1:numClasses
          if numGroups ~= numClasses
            index = i ;
          else
            index = 1 ;
          end
          weights(:,:,index,i) = (ones(1,k) - abs(C-center)./factor)'*(ones(1,k) - abs(C-center)./(factor));
        end          
      otherwise
        error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
    end
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
end
