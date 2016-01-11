% --------------------------------------------------------------------
function imdb = setupImdb_toyotaThumbs(opts)
% --------------------------------------------------------------------
dataDir = '/esat/malachite/bdebraba/Toyota/data/L/' ;

files = dir(fullfile(dataDir, '*.ppm'));
files = {files.name};
files = strcat(dataDir, files);
shrink = 16; 

nFiles = numel(files);
data = zeros(32,32,3,nFiles, 'single');
parfor i = 1:nFiles
    I = imread(files{i});
    
%     % make grayscale
%     I = rgb2gray(I);
    
    % crop to 512 by 512
    I = I(129:129+512-1, 257:257+512-1, :);
    
    % resize to 32 by 32
    I = imresize(I, 1/shrink);
    data(:,:,:,i) = I;
    i
end    

% rescale between -1 and 1
data = (data / 255)*2 - 1;

imdb.images.image = data ;
% imdb.images.label = cat(2, y1, y2) ;
imdb.images.set = ones(nFiles,1,'single') ;