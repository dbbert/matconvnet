function imdb = setupImdb_generated(opts)
opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/adversarial/data/mnist-6-test/net-epoch-60.mat';
% opts.netFile = '/users/visics/bdebraba/devel/matconvnet/examples/adversarial/data/toyotaThumbs-1-test/net-epoch-11.mat';

nets = load(opts.netFile);
generatorNet = dagnn.DagNN.loadobj(nets.generatorNet) ;
generatorNet.move('gpu') ;
generatorNet.mode = 'test';
vectorSize = 50;

imdb.images.image = zeros(32,32,1,0,'single') ;
imdb.images.label = zeros(1,1,vectorSize,0,'single') ;

for i = 1:256
    randomVectors = randn(1,1,vectorSize,256,'single') ;
    randomVectors = vl_nnnormalizelp(randomVectors, [], 'epsilon', eps) ;

    generatorNet.eval({'input', gpuArray(randomVectors)});
    data = gather(generatorNet.vars(end).value);

    % % visualize
    % tiledImage = tileImage(results,16,2);
    % imshow(tiledImage);

    imdb.images.image = cat(4, imdb.images.image, data) ;
    imdb.images.label = cat(4, imdb.images.label, randomVectors) ;
end

nImages = size(imdb.images.label, 4) ;
imdb.images.set = ones(1,nImages) ;
imdb.images.set(3*end/4+1:end) = 2 ;