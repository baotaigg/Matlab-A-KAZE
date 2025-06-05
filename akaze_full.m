function [keypoints, descriptors, responseMaps] = akaze_full(grayImg)
    if size(grayImg, 3) == 3
        grayImg = rgb2gray(grayImg); 
    end


    gray = im2double(grayImg);
    [H, W] = size(gray);

    numLevels = 5;
    sigmas = linspace(1.0, 3.0, numLevels);
    responseMaps = cell(numLevels, 1);
    for i = 1:numLevels
        iter = round(sigmas(i) * 5);
        responseMaps{i} = perona_malik(gray * 255, iter, 20, 0.2);
    end

    hessianResp = cell(numLevels, 1);
    for i = 1:numLevels
        L = responseMaps{i};
        [Lx, Ly] = gradient(double(L));
        [Lxx, ~] = gradient(Lx);
        [~, Lyy] = gradient(Ly);
        hessianResp{i} = abs(Lxx .* Lyy - (gradient(Lx, 2)).^2);
    end

    all_locs = [];
    all_vals = [];
    all_orients = [];
    for i = 1:numLevels
        [vals, locs] = max_local_peaks(hessianResp{i});
        for j = 1:size(locs,1)
            y = locs(j,2); x = locs(j,1);
            patch = get_patch(gray, x, y, 19); 
            [dx, dy] = gradient(patch);
            angle = atan2(sum(dy(:)), sum(dx(:)));
            all_locs = [all_locs; locs(j,:)];
            all_vals = [all_vals; vals(j)];
            all_orients = [all_orients; angle];
        end
    end

    K = 300;
    [~, sortIdx] = sort(all_vals, 'descend');
    sortIdx = sortIdx(1:min(K, length(sortIdx)));
    keypoints = all_locs(sortIdx, :);
    orientations = all_orients(sortIdx);

    patchSize = 31;
    gridSize = 4; % 4x4 = 16 cells
    descriptors = zeros(K, gridSize * gridSize);
    padded = padarray(gray, [patchSize patchSize], 'replicate');

    for i = 1:K
        y = keypoints(i,2); x = keypoints(i,1); theta = orientations(i);
        patch = padded(y:y+patchSize-1, x:x+patchSize-1);
        patch = rotate_patch(patch, -theta);
        small = imresize(patch, [gridSize, gridSize]);
        descriptors(i,:) = small(:)' > mean(small(:));
    end
end

function patch = get_patch(img, x, y, size)
    half = floor(size/2);
    padded = padarray(img, [half half], 'replicate');
    patch = padded(y:y+size-1, x:x+size-1);
end

function rot_patch = rotate_patch(patch, theta)
    tform = affine2d([cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1]);
    center = floor(size(patch)/2);
    R = imref2d(size(patch), [-center(2), center(2)], [-center(1), center(1)]);
    rot_patch = imwarp(patch, tform, 'OutputView', R, 'Interpolation', 'bilinear');
end

function [val, locs] = max_local_peaks(img)
    val = [];
    locs = [];
    sz = size(img);
    for i = 2:sz(1)-1
        for j = 2:sz(2)-1
            patch = img(i-1:i+1, j-1:j+1);
            if img(i,j) == max(patch(:)) && img(i,j) > 0.01
                val(end+1) = img(i,j);
                locs(end+1,:) = [j, i];
            end
        end
    end
end

function img_out = perona_malik(img, n_iter, kappa, lambda)
    img = double(img);
    [rows, cols] = size(img);
    img_out = img;
    for t = 1:n_iter
        north = zeros(rows, cols); north(2:end,:) = img_out(1:end-1,:);
        south = zeros(rows, cols); south(1:end-1,:) = img_out(2:end,:);
        west  = zeros(rows, cols); west(:,2:end) = img_out(:,1:end-1);
        east  = zeros(rows, cols); east(:,1:end-1) = img_out(:,2:end);
        dn = north - img_out;
        ds = south - img_out;
        dw = west  - img_out;
        de = east  - img_out;
        cN = exp(-(dn / kappa).^2);
        cS = exp(-(ds / kappa).^2);
        cW = exp(-(dw / kappa).^2);
        cE = exp(-(de / kappa).^2);
        img_out = img_out + lambda * (cN.*dn + cS.*ds + cW.*dw + cE.*de);
    end
    img_out = uint8(img_out);
end
