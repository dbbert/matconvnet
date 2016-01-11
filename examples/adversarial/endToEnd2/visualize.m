function visualize(generatorNet)

if nargin == 0
    clear;
    opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/adversarial/endToEnd/data/mnist-1-test/net-epoch-11.mat';
%     opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/adversarial/data/mnist-6-test/net-epoch-60.mat';
    % opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/adversarial/data/toyotaThumbs-1-test/net-epoch-11.mat';

    nets = load(opts.netFile);
    generatorNet = dagnn.DagNN.loadobj(nets.generatorNet) ;
end
generatorNet.move('gpu') ;
generatorNet.mode = 'test';
vectorSize = 50;

%% display random generations
% randomVectors = gpuArray.rand(1,1,vectorSize,256,'single')*2-1 ;
rng(0);
randomVectors = gpuArray.randn(1,1,vectorSize,256,'single') ;

generatorNet.eval({'input', randomVectors});
results = gather(generatorNet.vars(end).value);
tiledImage = tileImage(results,16,2);
figure;
imshow(tiledImage);

%% interpolation
if nargin == 0
    results = zeros(16,16,1,256,'single');
    randomVector2 = randn(vectorSize,1) ;
    for i = 1:16
        randomVector1 = randomVector2 ;
        randomVector2 = randn(vectorSize,1) ;

        q = single(linspace(1,0,16));
        randomVectors = randomVector1 * q + randomVector2 * (1-q);
        randomVectors = gpuArray(permute(randomVectors, [3 4 1 2]));

        generatorNet.eval({'input', randomVectors});
        results(:,:,:,(i-1)*16+1:i*16) = gather(generatorNet.vars(end).value);
    end

    tiledImage = tileImage(results,16,2);
    figure;
    imshow(tiledImage);
end