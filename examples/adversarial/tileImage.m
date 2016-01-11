function tiledImage = tileImage(tiles, cols, border)
    tileDims = [size(tiles,1) size(tiles,2), size(tiles,3)];
    nTiles = size(tiles,4);
    rows = ceil(nTiles / cols);
    tiledImage = zeros(rows*(tileDims(1) + border), cols*(tileDims(2) + border), tileDims(3));
    for i = 1:nTiles
        [x,y] = ind2sub([cols,rows], i);
        x = (x-1)*(tileDims(2)+border)+1;
        y = (y-1)*(tileDims(1)+border)+1;
        filter = tiles(:,:,:,i);
%         tiledImage(y:y+tileDims(1)-1, x:x+tileDims(2)-1, :) = mat2gray(filter);
        tiledImage(y:y+tileDims(1)-1, x:x+tileDims(2)-1, :) = filter;
    end
    tiledImage = mat2gray(tiledImage);
end
