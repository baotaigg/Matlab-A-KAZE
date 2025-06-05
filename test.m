img = imread('cameraman.tif');
if size(img,3) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end
[keypoints, descriptors, maps] = akaze_full(img_gray);

[h, w, ~] = size(img);
min_dim = min(h, w);
marker_size = max(3, round(min_dim / 100));

imshow(img); hold on;
plot(keypoints(:,1), keypoints(:,2), 'ro', 'MarkerSize', marker_size, 'LineWidth', 1.5);
