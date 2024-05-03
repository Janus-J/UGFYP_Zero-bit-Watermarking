clc; clear; close all;

%% INPUT
rawimage = imread("blk1.pgm"); %Input original image
normalizedImage = mat2gray(rawimage); %Normalization
%normalizedImage = rgb2gray(normalizedImage); %Transfer rgb to gray

%If with color image
if size(normalizedImage, 3) == 3
    normalizedImage = rgb2gray(normalizedImage); 
end

%Show the rawimage and normalizedimage
imshow(rawimage);
title('Original Image');
figure;
imshow(normalizedImage);
title('Normalized Image');


%% SECURITY KEY GENERATION 
blockSize = 8; %Define block size(number)
[height, width] = size(normalizedImage); %Get image size

%Calculate number of blocks in both dimensions
numBlocksRows = floor(height / blockSize);
numBlocksCols = floor(width / blockSize);
numBlocks = numBlocksRows * numBlocksCols;

securitykey = zeros(blockSize*blockSize, 1); %Initialize securitykey matrix to store binary values

%Iterate through each block in the original image
for i = 1:blockSize
    for j = 1:blockSize
        
        currentBlock = normalizedImage((i-1)*numBlocksRows + 1:i*numBlocksRows, (j-1)*numBlocksCols + 1:j*numBlocksCols); %Extract current block from the original image
        
        [LL_block, ~, ~, ~] = dwt2(currentBlock, 'haar'); %Perform DWT on the current block
        
        [~, S, ~] = svd(LL_block); %Perform SVD on the LL subband of the block
        
        firstSingularValue = S(1, 1); %Get the first singular value of S matrix
        
        %Store binary value based on comparison with previous block
        if i == 1 && j == 1
            securitykey(1) = 0; %For the first block, store '0'(can also be '1')
        else

            %Compare with previous block's singular value
            prevSingularValue = S_prev(1, 1);
            if firstSingularValue > prevSingularValue
                securitykey((i-1)*blockSize + j) = 1;
            else
                securitykey((i-1)*blockSize + j) = 0;
            end

        end
        
        S_prev = S; %Store current block's singular value for comparison in the next iteration
    end
end


%% FINGERPRINT(ID PROCESSING)
fingerprintImage = imread('fp.jpg'); %input fingerprint image
threshold = graythresh(fingerprintImage); %otsu'method to get the threshold automatically
binaryFingerprint1 = imbinarize(fingerprintImage, threshold); %Binarization
binaryFingerprint1 = fingerprintImage > threshold;

%Get the size of binaryFingerprint
binaryFingerprint=rgb2gray(uint8(binaryFingerprint1)); %Transfer to the grayscale one
[rows, cols] = size(binaryFingerprint);
num=numel(securitykey);

%Cover the securitykey to the whole image, for the XOR operation
k = 1;
r = 1;
c = 1;
    while c < (cols + 1)
        
            while k < num + 1 && r < (rows + 1)
                securitykeyMatrix(r,c) = securitykey(k);
                k = k + 1;
                r = r + 1;
            end
            
            if k == num + 1
               k = 1;
            end

            if r == rows + 1
               c = c + 1;
               r = 1;
            end
    end

%XOR two same sizes matrix to get the watermark
watermark = xor(securitykeyMatrix,binaryFingerprint);

%Store to the database in PNG form
imwrite(uint8(watermark)*255, 'watermark.png');

%Read the image data from it
wmimage = imread('watermark.png');

%Show the watermark
figure;
imshow(watermark);
title('Generated Watermark');

%% Store to the database
%Database file position
databaseFile = 'imageDatabase.mat';
%Add the original image and watermark to the database
addImageToDatabase(rawimage, watermark, databaseFile);

%%
function addImageToDatabase(originalImage, watermarkImage, databaseFile)
    %Check if the database file is existed
    if exist(databaseFile, 'file')
        %Load the database
        load(databaseFile, 'imageDatabase');
    else
        %Create a new database
        imageDatabase = struct('num', {}, 'original', {}, 'watermark', {}, 'timestamp', {});
    end

    %Calculate the new num
    if isempty(imageDatabase)
        newnum = 1;  %If the database is empty, start from '1'
    else
        newnum = max([imageDatabase.num]) + 1;  %Otherwise, num=currentMaxValue+1
    end

    %Same images can not be stored
    isNewUnique = true;
    for i = 1:length(imageDatabase)
        if isequal(originalImage, imageDatabase(i).original)
            isNewUnique = false;
            disp('Image already exists in the database.');
            break;
        end
    end

    if isNewUnique
    %Add the new image information with the timestamp
    entry.num = newnum;
    entry.original = originalImage;
    entry.watermark = watermarkImage;
    entry.timestamp = datetime('now');
    %Add the new item into the database
    imageDatabase(end+1) = entry;
    %Update the database
    save(databaseFile, 'imageDatabase');
    %Tell the user the 'num' of his image
    fprintf('Your Image Index is: %d\n', newnum);
    end
    
end

