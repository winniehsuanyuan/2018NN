clc
clear;
%load train data
filename1_1 = '/Users/Winnie/Downloads/三上/圖形識別/matlabprac/train-images.idx3-ubyte';
fp1_1=fopen(filename1_1,'r');
magic = fread(fp1_1, 1, 'int32', 0, 'ieee-be');
ntrain = fread(fp1_1, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp1_1, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp1_1, 1, 'int32', 0, 'ieee-be');
rawImgDataTrain = uint8(fread(fp1_1, ntrain * numRows * numCols, 'uint8'));
fclose(fp1_1);
% Reprocess the data part into a 4D array
rawImgDataTrain = reshape(rawImgDataTrain, [numRows, numCols, ntrain]);
rawImgDataTrain = permute(rawImgDataTrain, [2,1,3]);
imgDataTrain(:,:,1,:) = uint8(rawImgDataTrain(:,:,:));
%train label
filename1_2 = '/Users/Winnie/Downloads/三上/圖形識別/matlabprac/train-labels.idx1-ubyte';
fp1_2=fopen(filename1_2,'r');
magic = fread(fp1_2, 1, 'int32', 0, 'ieee-be');
ntrain = fread(fp1_2, 1, 'int32', 0, 'ieee-be');
labelsTrain = fread(fp1_2, ntrain, 'uint8');
fclose(fp1_2);
labelsTrain = categorical(labelsTrain);

%load test data
filename2_1 = '/Users/Winnie/Downloads/三上/圖形識別/matlabprac/t10k-images.idx3-ubyte';
fp2_1=fopen(filename2_1,'r');
magic = fread(fp2_1, 1, 'int32', 0, 'ieee-be');
ntest = fread(fp2_1, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp2_1, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp2_1, 1, 'int32', 0, 'ieee-be');
rawImgDataTest = uint8(fread(fp2_1, ntest * numRows * numCols, 'uint8'));
fclose(fp2_1);
% Reprocess the data part into a 4D array
rawImgDataTest = reshape(rawImgDataTest, [numRows, numCols, ntest]);
rawImgDataTest = permute(rawImgDataTest, [2,1,3]);
imgDataTest = uint8(zeros(numRows, numCols, 1, ntest));
imgDataTest(:,:,1,:) = uint8(rawImgDataTest(:,:,:));
%test label
filename2_2 = '/Users/Winnie/Downloads/三上/圖形識別/matlabprac/t10k-labels.idx1-ubyte';
fp2_2=fopen(filename2_2,'r');
magic = fread(fp2_2, 1, 'int32', 0, 'ieee-be');
ntest = fread(fp2_2, 1, 'int32', 0, 'ieee-be');
labelsTest = fread(fp2_2, ntest, 'uint8');
fclose(fp2_2);
labelsTest = categorical(labelsTest);

layers = [
    imageInputLayer([28 28 1], 'Name', 'input')
	
    convolution2dLayer(3,16,'Padding',1, 'Name', 'conv_1')
    batchNormalizationLayer('Name', 'BN_1')
    reluLayer('Name', 'relu_1')
	
    maxPooling2dLayer(2,'Stride',2,'Name', 'MP_1')
	
    convolution2dLayer(3,32,'Padding',1,'Name', 'conv_2')
    batchNormalizationLayer('Name', 'BN_2')
    reluLayer('Name', 'relu_2')
	
    maxPooling2dLayer(2,'Stride',2,'Name', 'MP_2')
	
    convolution2dLayer(3,64,'Padding',1, 'Name', 'conv_3')
    batchNormalizationLayer('Name', 'BN_3')
    reluLayer('Name', 'relu_3')
	
    fullyConnectedLayer(10, 'Name', 'FC_1')
    softmaxLayer('Name', 'SM')
    classificationLayer('Name', 'output')];
miniBatchSize = 8192;
options = trainingOptions( 'sgdm',...
    'MiniBatchSize', miniBatchSize,...
    'Plots', 'training-progress');
%%%plot layers
lgraph=layerGraph(layers);
figure;
plot(lgraph);

%%%1000 train
net1 = trainNetwork(imgDataTrain(:,:,1,1:1000), labelsTrain(1:1000), layers, options);
%%%1000 test
predLabelsTest1 = net1.classify(imgDataTest(:,:,1,1:1000));
accuracy = sum(predLabelsTest1 == labelsTest(1:1000)) / 1000
%150 testing patterns 
figure;
hold on;
target=zeros(150, 1);

for i = 1:150
      subplot(15, 10, i);
      img=imgDataTest(:,:,1,i);
      imshow(img);
      title(char(labelsTest(i)));
      text(0, 35, sprintf('predicted: %c',predLabelsTest1(i)));
end
%confusion
figure;
plotconfusion(labelsTest(1:150), predLabelsTest1(1:150));

%plot feature map
for layer = 2:15
    name = net1.Layers(layer).Name;
    if strfind(name, "conv")==1
        channels=net1.Layers(layer).NumFilters;
    else if strfind(name, "FC")==1
            channels=net1.Layers(layer).OutputSize;
        end
    end
    I = deepDreamImage(net1,layer,1: channels, 'Verbose',false,'PyramidLevels',1);
    figure;
    I = imtile(I,'ThumbnailSize',[64 64]);
    imshow(I);
    title(['Layer ',name,' Features'])
end

%%%%%%%%%%%%%%
%%%60000 train
net2 = trainNetwork(imgDataTrain, labelsTrain, layers, options);
%%%10000 test
predLabelsTest2 = net2.classify(imgDataTest);
accuracy = sum(predLabelsTest2 == labelsTest) / numel(labelsTest)
%150 testing patterns 
figure;
hold on;
target=zeros(150, 1);

for i = 1:150
      subplot(15, 10, i);
      img=imgDataTest(:,:,1,i);
      imshow(img);
      title(char(labelsTest(i)));
      text(0, 35, sprintf('predicted: %c',predLabelsTest2(i)));
end
%confusion
figure;
plotconfusion(labelsTest(1:150), predLabelsTest2(1:150));
