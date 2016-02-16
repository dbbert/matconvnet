% -------------------------------------------------------------------------
function net = getNet(varargin)
% -------------------------------------------------------------------------
    opts.type = 'deepTracking'; % or discriminator
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
    
    net = dagnn.DagNN() ;
    
    % encoder
% %     outputs = add_conv(net, netOpts, 'block1', inputs, 16, 16, netOpts.nChns, 10, 1, 0);
%     outputs = add_conv(net, netOpts, 'block1', inputs, 7, 7, netOpts.nChns, 32, 1, 3);
%     outputs = add_normalizeL1(net, netOpts, 'normalize', outputs);
%     net.addLayer('repellingloss', RepellingLoss('loss', 'repelling'), outputs, 'objective') ;

    vectorSize = 50;

    switch opts.type
        case 'deepTracking'
            
            nLoops = 4; % last image should be empty
%             previousBelief = 'initialBelief';
            previousBelief = add_conv(net, netOpts, 'initialBelief', 'zeroImage', 1, 1, 1, 8, 1, 0, true); % has bias
            for i = 1:nLoops
                
                if i < nLoops
                    inputImage = {sprintf('image%d', i)} ;
                else
                    inputImage = 'zeroImage' ;
                end
                
                % encode
                outputs = add_block(net, netOpts, sprintf('block1_%03d', i), inputImage, 3, 3, netOpts.nChns, 8, 1, 1);
                outputs = add_block(net, netOpts, sprintf('block1b_%03d', i), outputs, 3, 3, 8, 8, 1, 1);
                
%                 outputs = add_conv(net, netOpts, sprintf('block1_%03d', i), inputImage, 7, 7, netOpts.nChns, 8, 1, 3);
%                 outputs = add_batchNorm(net, netOpts, sprintf('block1_%03d', i), outputs, 8);
%                 outputs = add_relu(net, netOpts, sprintf('block1_%03d', i), outputs) ;

                % concatenate with previous belief and calculate the
                % updated belief
                outputs = add_concat(net, netOpts, sprintf('block2_%03d', i), [outputs, previousBelief]);
                outputs = add_block(net, netOpts, sprintf('block2_%03d', i), outputs, 3, 3, 16, 16, 1, 1);
                updatedBelief = add_block(net, netOpts, sprintf('block2b_%03d', i), outputs, 3, 3, 16, 8, 1, 1);
                
%                 outputs = add_conv(net, netOpts, sprintf('block2_%03d', i), outputs, 5, 5, 16, 8, 1, 2);
%                 outputs = add_batchNorm(net, netOpts, sprintf('block2_%03d', i), outputs, 8);
%                 updatedBelief = add_relu(net, netOpts, sprintf('block2_%03d', i), outputs);
                
                previousBelief = updatedBelief;
            end
            
            % decode
            outputs = add_deblock(net, netOpts, sprintf('block3_%03d', nLoops), updatedBelief, 3, 3, 8, 8, 1, 1);
            outputs = add_conv(net, netOpts, sprintf('block3b_%03d', nLoops), outputs, 3, 3, 8, 2, 1, 1);
            
%             outputs = add_conv(net, netOpts, sprintf('block3_%03d', nLoops), updatedBelief, 7, 7, 8, 2, 1, 3);
% %             outputs = add_tanh(net, netOpts, sprintf('block3_%03d', nLoops), outputs) ;
            
            net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), [outputs, {sprintf('image%d', nLoops)}], 'objective') ;
            
        case 'deepTrackingConcat'
            outputs = add_concat(net, netOpts, 'block0', {'image1','image2','image3'});
            outputs = add_block(net, netOpts, 'block1', outputs, 3, 3, netOpts.nChns * 3, 8, 1, 1);
            outputs = add_block(net, netOpts, 'block2', outputs, 3, 3, 8, 32, 1, 1);
            outputs = add_block(net, netOpts, 'block3', outputs, 3, 3, 32, 8, 1, 1);
            outputs = add_conv(net, netOpts, 'block4', outputs, 3, 3, 8, 2, 1, 1);
            net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), [outputs, {'image4'}], 'objective') ;
            
        case 'default'
            outputs = add_conv(net, netOpts, 'block1', {'image'}, 16, 16, netOpts.nChns, 256, 1, 0);
%             outputs = add_batchNorm(net, netOpts, 'block1', outputs, 512);
            outputs = add_sigmoid(net, netOpts, 'block1', outputs) ;
            
            outputs = add_conv(net, netOpts, 'block2', outputs, 1, 1, 256, 128, 1, 0);
%             outputs = add_batchNorm(net, netOpts, 'block2', outputs, 128);
            outputs = add_sigmoid(net, netOpts, 'block2', outputs) ;
            
            net.addLayer('lossMmd', MmdLoss(), [outputs, {'inputNoise'}], 'mmd') ;
            
            outputs = add_deconv(net, netOpts, 'block2_ab', outputs, 1, 1, 128, 256, 1, 0);
%             outputs = add_batchNorm(net, netOpts, 'deblock2', outputs, 512);
            outputs = add_sigmoid(net, netOpts, 'deblock2', outputs) ;
            
            outputs = add_deconv(net, netOpts, 'block1_ab', outputs, 16, 16, 256, netOpts.nChns, 1, 0);
            outputs = add_tanh(net, netOpts, 'deblock1', outputs) ; % this might not work very well in combination with repelling loss
            net.addLayer('loss', dagnn.Loss('loss', 'L1'), [outputs, {'image'}], 'objective') ;
            
        case 'default2'
            % encoder
            outputs = add_conv(net, netOpts, 'block1', {'image'}, 4, 4, netOpts.nChns, 16, 2, 1);
            outputs = add_batchNorm(net, netOpts, 'block1', outputs, 16);
            outputs = add_sigmoid(net, netOpts, 'block1', outputs) ;
            outputs = add_dropout(net, netOpts, 'block1', outputs, 0.25);            
       
            outputs = add_conv(net, netOpts, 'block2', outputs, 4, 4, 16, 32, 2, 1);
            outputs = add_batchNorm(net, netOpts, 'block2', outputs, 32);
            outputs = add_sigmoid(net, netOpts, 'block2', outputs) ;
            outputs = add_dropout(net, netOpts, 'block2', outputs, 0.25);

            outputs = add_conv(net, netOpts, 'block3', outputs, 4, 4, 32, 64, 2, 1);
            outputs = add_batchNorm(net, netOpts, 'block3', outputs, 64);
            outputs = add_sigmoid(net, netOpts, 'block3', outputs) ;
            outputs = add_dropout(net, netOpts, 'block3', outputs, 0.25);

            outputs = add_conv(net, netOpts, 'block4', outputs, 4, 4, 64, 128, 1, 0);
            outputs = add_batchNorm(net, netOpts, 'block4', outputs, 128);
            outputsX = add_sigmoid(net, netOpts, 'block4', outputs) ;
            outputs = add_dropout(net, netOpts, 'block4', outputsX, 0.25);

% %             outputs = add_conv(net, netOpts, 'block5', outputs, 1, 1, 128, vectorSize, 1, 0);
% % %             outputs = add_batchNorm(net, netOpts, 'block5', outputs, vectorSize);
% % %             outputs = add_relu(net, netOpts, 'block5', outputs) ;
% % %             outputs = add_dropout(net, netOpts, 'block5', outputs, 0.25);
% 
%             % decoder
% %             outputs = add_normalizeLp(net, netOpts, 'normalizeLp', outputs, 2);

%             outputs2 = add_conv(net, netOpts, 'blockNoise1', {'inputNoise'}, 1, 1, 64, 64, 1, 0);
%             outputs2 = add_batchNorm(net, netOpts, 'blockNoise1', outputs2, 64);
%             outputs2 = add_sigmoid(net, netOpts, 'blockNoise1', outputs2) ;
%             outputs2 = add_conv(net, netOpts, 'blockNoise2', outputs2, 1, 1, 64, 128, 1, 0);
%             outputs2 = add_batchNorm(net, netOpts, 'blockNoise2', outputs2, 128);
%             outputs2 = add_sigmoid(net, netOpts, 'blockNoise2', outputs2) ;
% %             outputs2 = add_normalizeLp(net, netOpts, 'blockNoise2', outputs2, 2);
            net.addLayer('lossMmd', MmdLoss(), [outputsX, {'inputNoise'}], 'mmd') ;
            
% %             outputs = add_deconv(net, netOpts, 'block5_abc', outputs, 1, 1, vectorSize, 128, 1, 0);
% %             outputs = add_batchNorm(net, netOpts, 'deblock5', outputs, 128);
% %             outputs = add_relu(net, netOpts, 'deblock5', outputs) ;
            
            outputs = add_deconv(net, netOpts, 'block4_abc', outputs, 4, 4, 128, 64, 1, 0);
            outputs = add_batchNorm(net, netOpts, 'deblock4', outputs, 64);
            outputs = add_sigmoid(net, netOpts, 'deblock4', outputs) ;

            outputs = add_deconv(net, netOpts, 'block3_abc', outputs, 4, 4, 64, 32, 2, 1);
            outputs = add_batchNorm(net, netOpts, 'deblock3', outputs, 32);
            outputs = add_sigmoid(net, netOpts, 'deblock3', outputs) ;

            outputs = add_deconv(net, netOpts, 'block2_abc', outputs, 4, 4, 32, 16, 2, 1);
            outputs = add_batchNorm(net, netOpts, 'deblock2', outputs, 16);
            outputs = add_sigmoid(net, netOpts, 'deblock2', outputs) ;           
            
            outputs = add_deconv(net, netOpts, 'block1_abc', outputs, 4, 4, 16, netOpts.nChns, 2, 1);
            outputs = add_tanh(net, netOpts, 'deblock1', outputs) ; % this might not work very well in combination with repelling loss
            net.addLayer('loss', dagnn.Loss('loss', 'L1'), [outputs, {'image'}], 'objective') ;
            
            pretrainedNetPath = '/users/visics/bdebraba/devel/matconvnet/examples/mnistAutoencoder/data/mnist-63-test/net-epoch-3.mat' ;
            net = replaceParams(net, pretrainedNetPath, false) ;
%             net = restoreParams(net) ;
            
        case 'encoder'
            %     outputs = add_dropout(net, netOpts, 'dropout0', inputs, 0.25);
            outputs = add_conv(net, netOpts, 'block1', {'image'}, 4, 4, netOpts.nChns, 32, 2, 1);
%             outputs = add_batchNorm(net, netOpts, 'block1', outputs, 32);
            outputs = add_relu(net, netOpts, 'block1', outputs) ;
       
            outputs = add_conv(net, netOpts, 'block2', outputs, 4, 4, 32, 64, 2, 1);
%             outputs = add_batchNorm(net, netOpts, 'block2', outputs, 64);
            outputs = add_relu(net, netOpts, 'block2', outputs) ;

            outputs = add_conv(net, netOpts, 'block3', outputs, 4, 4, 64, 128, 2, 1);
%             outputs = add_batchNorm(net, netOpts, 'block3', outputs, 128);
            outputs = add_relu(net, netOpts, 'block3', outputs) ;

            outputs = add_conv(net, netOpts, 'block4', outputs, 4, 4, 128, vectorSize, 1, 0);
        % %     % these two should be merged into one layer.    
        %     outputs = add_normalizeLp(net, netOpts, 'normalizeL1', outputs, 1);
        %     net.addLayer('repellingloss', RepellingLoss(), outputs, 'repel') ;
        case 'decoder'
%             outputs = add_normalizeLp(net, netOpts, 'normalizeLp', {'encoding'}, 2);
            outputs = add_deconv(net, netOpts, 'block4_a', {'encoding'}, 4, 4, vectorSize, 128, 1, 0);
            outputs = add_batchNorm(net, netOpts, 'deblock4', outputs, 128);
            outputs = add_relu(net, netOpts, 'deblock4', outputs) ;

            outputs = add_deconv(net, netOpts, 'block3_a', outputs, 4, 4, 128, 64, 2, 1);
            outputs = add_batchNorm(net, netOpts, 'deblock3', outputs, 64);
            outputs = add_relu(net, netOpts, 'deblock3', outputs) ;

            outputs = add_deconv(net, netOpts, 'block2_a', outputs, 4, 4, 64, 32, 2, 1);
            outputs = add_batchNorm(net, netOpts, 'deblock2', outputs, 32);
            outputs = add_relu(net, netOpts, 'deblock2', outputs) ;
            
            outputs = add_deconv(net, netOpts, 'block1_a', outputs, 4, 4, 32, netOpts.nChns, 2, 1);
%             outputs = add_tanh(net, netOpts, 'deblock1', outputs) ; % this might not work very well in combination with repelling loss
            
            net.addLayer('loss', dagnn.Loss('loss', 'L1'), [outputs, {'image'}], 'objective') ;
    end    
end

function net = restoreParams(net)
    for i = 1:numel(net.params)
        if isfield(net.params(i), 'learningRateOriginal')
            net.params(i).learningRate = net.params(i).learningRateOriginal;
        end
        if isfield(net.params(i), 'weightDecayOriginal')
            net.params(i).weightDecay = net.params(i).weightDecayOriginal;
        end
    end
end

function net = replaceParams(net, pretrainedNetPath, fixWeights)
    pretrainedNet = load(pretrainedNetPath);
    pretrainedNet = dagnn.DagNN.loadobj(pretrainedNet.net) ;
    
    for i = 1:numel(net.params)
        pretrainedParamIndex = pretrainedNet.getParamIndex(net.params(i).name) ;
        if ~isnan(pretrainedParamIndex)
            pretrainedParam = pretrainedNet.params(pretrainedParamIndex);
            net.params(i).value = pretrainedParam.value ;
            if fixWeights
                if isfield(net.params(i), 'learningRate')
                    if ~isfield(net.params(i), 'learningRateOriginal')
                        net.params(i).learningRateOriginal = net.params(i).learningRate;
                    end
                    net.params(i).learningRate = 0;
                end
                if isfield(net.params(i), 'weightDecay')
                    if ~isfield(net.params(i), 'weightDecay')
                        net.params(i).weightDecayOriginal = net.params(i).weightDecay;
                    end
                    net.params(i).weightDecay = 0;
                end
            end
        end
    end
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
    h = 2; w = 2;
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
      'pad', 0, ...
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
    outputs = add_conv(net, opts, name, inputs, h, w, in, out, stride, pad);
    outputs = add_batchNorm(net, opts, name, outputs, out);
    outputs = add_relu(net, opts, name, outputs, 0.2);
end

function outputs = add_blockNoBatchNorm(net, opts, name, inputs, h, w, in, out, stride, pad)
    outputsConv = add_conv(net, opts, name, inputs, h, w, in, out, stride, pad);
%     outputsBn = add_batchNorm(net, opts, name, outputsConv, out);
    outputsRelu = add_relu(net, opts, name, outputsConv, 0.2);
    
    outputs = outputsRelu;
end

function outputs = add_deblock(net, opts, name, inputs, h, w, in, out, stride, pad)
    outputs = add_deconv(net, opts, name, inputs, h, w, in, out, stride, pad);
    outputs = add_batchNorm(net, opts, name, outputs, out);
    outputs = add_relu(net, opts, name, outputs);
end

function outputs = add_conv(net, opts, name, inputs, h, w, in, out, stride, pad, hasBias)
    if nargin < 11
        hasBias = false;
    end
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

    learningRate = [1 2] ;
    weightDecay = [opts.weightDecay 0] ;
    
    params(1).name = sprintf('%sf',varName) ;
    params(1).value = init_weight(opts, h, w, in, out, 'single') ;
    params(1).learningRate = learningRate(1) ;
    params(1).weightDecay = weightDecay(1) ;
    
    if hasBias
        params(2).name = sprintf('%sb',varName) ;
        params(2).value = zeros(out, 1, 'single') ;
        params(2).learningRate = learningRate(2) ;
        params(2).weightDecay = weightDecay(2) ;
    end

    block = dagnn.Conv() ;
    block.size = size(params(1).value) ;
    block.pad = pad ;
    block.stride = stride ;
    block.hasBias = hasBias ;

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

function outputs = add_deconv(net, opts, name, inputs, h, w, in, out, upsample, crop, hasBias)
%     opts.weightInitMethod = 'bilinear';
    if nargin < 11
        hasBias = false;
    end
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
    
    learningRate = [1 2] ;
    weightDecay = [opts.weightDecay 0] ;

    params(1).name = sprintf('%sf',varName) ;
    params(1).value = init_weight(opts, h, w, out, in, 'single') ;
    params(1).learningRate = learningRate(1) ;
    params(1).weightDecay = weightDecay(1) ;
    
    if hasBias
        params(2).name = sprintf('%sb',varName) ;
        params(2).value = zeros(1, out, 'single') ;    
        params(2).learningRate = learningRate(2) ;    
        params(2).weightDecay = weightDecay(2) ;
    end

    block = dagnn.ConvTranspose() ;
    block.size = size(params(1).value) ;
    if numel(block.size) == 3
        block.size = [block.size 1];
    end
    block.upsample = upsample ;
    block.crop = crop ;
    block.hasBias = hasBias ;
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

function outputs = add_normalizeLp(net, opts, name, inputs, p)
    block = dagnn.NormalizeLp('p', p) ;

    layerName = sprintf('%s_normL1', name);
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
