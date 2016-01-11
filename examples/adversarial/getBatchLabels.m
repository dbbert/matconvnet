% -------------------------------------------------------------------------
function inputs = getBatchLabels(imdb, batch, opts)
% -------------------------------------------------------------------------
im = imdb.images.image(:,:,:,batch);
labels = imdb.images.label(:,:,:,batch);

netInputs = {};
if opts.useGpu
    netInputs = [netInputs {'input', gpuArray(im)}];
    netInputs = [netInputs {'label', gpuArray(labels)}];
else
    netInputs = [netInputs {'input', im}];
    netInputs = [netInputs {'label', labels}];
end

inputs = netInputs;
end