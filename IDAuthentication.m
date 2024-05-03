clc; clear; close all;

%% INPUT
%Database file position
databaseFile = 'imageDatabase.mat';
%Import the original image and the watermark from the database
[OriginalImage, WatermarkImage, timestampRetrieved] = getImageFromDatabase(3, databaseFile);
disp(['Storage Time: ', datestr(timestampRetrieved)]);

BinaryWatermark = WatermarkImage/255; %Binarization
OriginalHist = imhist(OriginalImage); %Get the standard histogram for correction
%OriginalImage = rgb2gray(OriginalImage); %If with color image

%If with color image
if size(OriginalImage, 3) == 3
    OriginalImage = rgb2gray(OriginalImage); 
end

%% Simulate the attacks
%Multiple JPEG compressions attack
jpegQuality = 90;
imwrite(OriginalImage, '1stJPEGcompressedImage.jpg', 'Quality', jpegQuality);
CompressedImage1 = imread("1stJPEGcompressedImage.jpg");
jpegQuality = 85; 
imwrite(CompressedImage1, '2ndJPEGcompressedImage.jpg', 'Quality', jpegQuality);
CompressedImage2 = imread("2ndJPEGcompressedImage.jpg");

%Rotation attack
RotatedImage = imrotate(CompressedImage2, 3, 'bicubic', 'loose');

%Scaling to 0.95
Jc_Ro_Sca_Image = imresize(RotatedImage, 0.95);

%Nongeometric attacks
NongeometricImage = imadjust(Jc_Ro_Sca_Image, [0.1, 0.9],[0.3, 0.7]);

%3D-rotation attacks
tform = projective2d([1 0.025 0; 0.025 1 0; 0.0015 0.0015 1]); %Transfer array example
AttackedImage = imwarp(NongeometricImage, tform);

figure;
%Show the original image
subplot(1,2,1);
imshow(OriginalImage);
title('Original Image');
%Show the attacked image
subplot(1,2,2);
imshow(AttackedImage);
title('Attacked Image');

%% Robust to geometric attacks(rotation, scaling)
%Using SIFT features to correct the attacked image through comparsion
ptsOriginal  = detectSIFTFeatures(OriginalImage);
ptsDistorted = detectSIFTFeatures(AttackedImage);
[featuresOriginal, validPtsOriginal] = extractFeatures(OriginalImage, ptsOriginal);
[featuresDistorted, validPtsDistorted] = extractFeatures(AttackedImage, ptsDistorted);

indexPairs = matchFeatures(featuresOriginal, featuresDistorted);
matchedPtsOriginal  = validPtsOriginal(indexPairs(:, 1));
matchedPtsDistorted = validPtsDistorted(indexPairs(:, 2));

[tform, inlierIdx] = estgeotform2d(matchedPtsDistorted, matchedPtsOriginal, 'similarity');
inlierPtsDistorted = matchedPtsDistorted(inlierIdx, :);
inlierPtsOriginal  = matchedPtsOriginal(inlierIdx, :);

outputView = imref2d(size(OriginalImage));
CorrectedImage = imwarp(AttackedImage, tform, 'OutputView', outputView, 'interp', 'bicubic'); %Bicubic interpolation

%Show the corrected image
figure;
imshow(CorrectedImage);
title('Corrected Image');


%% For 3D-rotation, and correct the image
%Detect the feature points
pointsOriginal = detectSIFTFeatures(OriginalImage);
pointsDistorted = detectSIFTFeatures(CorrectedImage);

%Extract the features and match them
[featuresOriginal, validPointsOriginal] = extractFeatures(OriginalImage, pointsOriginal);
[featuresDistorted, validPointsDistorted] = extractFeatures(CorrectedImage, pointsDistorted);
indexPairs = matchFeatures(featuresOriginal, featuresDistorted);
matchedPointsOriginal = validPointsOriginal(indexPairs(:, 1), :);
matchedPointsDistorted = validPointsDistorted(indexPairs(:, 2), :);

%Visualizing the matching points
figure; showMatchedFeatures(OriginalImage, CorrectedImage, matchedPointsOriginal, matchedPointsDistorted, 'montage');

%The transformation matrix is estimated using the matching points
[tform, inlierDistorted, inlierOriginal] = estimateGeometricTransform(matchedPointsDistorted, matchedPointsOriginal, 'projective');
legend("Matched points 1","Matched points 2");

%The inverse transform is applied to correct the image
outputView = imref2d(size(OriginalImage));
recoveredImage = imwarp(CorrectedImage, tform, 'OutputView', outputView);

%Show the corrected image
figure; imshow(recoveredImage);
title('Corrected Image');


%% Robust to non-geometric attacks
% USM for correcting the original image details(sharpening)
recoveredImage = usm(recoveredImage, 1, 1, 7); %Have a function in the end
normalizedImage2 = mat2gray(recoveredImage); %Normalization
normalizedImage2 = histeq(normalizedImage2, OriginalHist); %Correct the histogram to the standard one

%Show the results
figure, imshow(normalizedImage2), title('Ready Image(sharpened)');


%% SECURITY KEY GENERATION 
blockSize = 8; %Define block size(number)
[height2, width2] = size(normalizedImage2); %Get image size

% Calculate number of blocks in both dimensions
numBlocksRows2 = floor(height2 / blockSize);
numBlocksCols2 = floor(width2/ blockSize);
numBlocks2 = numBlocksRows2 * numBlocksCols2;

binaryMatrix2 = zeros(blockSize*blockSize, 1); %Initialize securitykey matrix to store binary values

% Iterate through each block in the original image
for i = 1:blockSize
    for j = 1:blockSize
        
        currentBlock2 = normalizedImage2((i-1)*numBlocksRows2 + 1:i*numBlocksRows2, (j-1)*numBlocksCols2 + 1:j*numBlocksCols2); %Extract current block from the original image
        
        [LL_block, ~, ~, ~] = dwt2(currentBlock2, 'haar'); %Perform DWT on the current block
        
        [~, S, ~] = svd(LL_block); %Perform SVD on the LL component of the block
        
        firstSingularValue2 = S(1, 1); %Get the first singular value
        
        % Store binary value based on comparison with previous block
        if i == 1 && j == 1
            binaryMatrix2(1) = 0; %For the first block, store '0'(can also be '1')
        else

            %Compare with previous block's singular value
            prevSingularValue2 = S_prev(1, 1);
            if firstSingularValue2 > prevSingularValue2
                binaryMatrix2((i-1)*blockSize + j) = 1;
            else
                binaryMatrix2((i-1)*blockSize + j) = 0;
            end
        end
        
        S_prev = S; % Store current block's singular value for comparison in the next iteration
    end
end

num=numel(binaryMatrix2);

%Cover the securitykey to the same size zero-matrix, for the XOR operation
k = 1;
r = 1;
c = 1;
    while c < (644 + 1)
        
            while k < num + 1 && r < (900 + 1)
                securitykey2(r,c) = binaryMatrix2(k);
                k = k + 1;
                r = r + 1;
            end
            
            if k == num + 1
               k = 1;
            end

            if r == 900 + 1
               c = c + 1;
               r = 1;
            end
    end

%Extract the ID to compare with the fingerprint
fp2 = xor(securitykey2, BinaryWatermark);

%Save as image file(PNG form)
imwrite(uint8(fp2)*255, 'extract_fp.png');

%Read the image data
ID = imread('extract_fp.png');

%Show the extracted ID image
figure;
imshow(ID);
title('Extracted ID');

%% NCC
%% Read two binary images (make sure they are the same size)
%provided fingerprint
providefingerprint2 = imread('fp.jpg');
threshold = graythresh(providefingerprint2);
binaryFingerprint3 = imbinarize(providefingerprint2, threshold);
binaryFingerprint3 = providefingerprint2 > threshold;
binaryFingerprint4=rgb2gray(uint8(binaryFingerprint3));
providefingerprint3 = binaryFingerprint4*255;

%Calculate NCC value
ncc = normxcorr2(providefingerprint3, ID);
%Pop-up window to show the result
if max(ncc(:)) > 0.5
        msgbox('Successful Authentication!!!', 'RESULT', 'help');
else
        msgbox('Authentication Failure.', 'RESULT', 'error');
end

%Display NCC results (maximum value of the ncc matrix indicates the best match position)
disp(['NCC = ' num2str(max(ncc(:)))]);

%% USM FUNCTION
function I_sharpened = usm(I, radius, amount, threshold)
%Unsharp Masking/UNSHARPENING MASKS SHARPENING
%INPUT:
%I - INPUT GRAYSCALE IMAGE
%RADIUS - GAUSSIAN BLUR RADIUS(STANDARD DEVIATION)
%AMOUNT - INTENSITY OF SHARPENING
%THRESHOLD - SHARPENING THRESHOLD, PERFORMED ONLY ON THE DIFFERENCES ABOVE THE THRESHOLD
    I_blurred = imgaussfilt(I, radius); %Gaussian blur is applied to the original image
    I_diff = imsubtract(double(I), double(I_blurred)); %The difference image between the original image and the blurred image is calculated
    I_diff(abs(I_diff) < threshold) = 0; %A threshold is applied and only differences greater than the threshold are sharpened
    I_sharpened = double(I) + (amount * I_diff); %The weighted difference image is added back to the original image to enhance the details %Make sure have the same types
    I_sharpened = im2uint8(mat2gray(I_sharpened)); %The result is scaled and typecast to match the original image type
end

%%
function [originalImage, watermarkImage, timestamp] = getImageFromDatabase(index, databaseFile)
    load(databaseFile, 'imageDatabase');
    if index > length(imageDatabase) || index < 1
        error('Index out of bounds.');
    end
    originalImage = imageDatabase(index).original;
    watermarkImage = imageDatabase(index).watermark;
    timestamp = imageDatabase(index).timestamp;
end

