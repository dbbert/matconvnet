% --------------------------------------------------------------------
function imdb = setupImdb_test(opts)
% --------------------------------------------------------------------
nImages = 1024*50;

imdb.images.image = single(repmat([1 1 1; 1 -1 1; -1 -1 -1], 1,1,1,nImages));
imdb.images.label = ones(1,1,1,nImages, 'single');
imdb.images.set = ones(1,nImages, 'single');
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
