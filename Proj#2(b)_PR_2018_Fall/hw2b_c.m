clear;
clc
%load train data
filename1_1 = '/Users/Winnie/Downloads/三上/圖形識別/matlabprac/train-images.idx3-ubyte';
fp1_1=fopen(filename1_1,'r');
magic = fread(fp1_1, 1, 'int32', 0, 'ieee-be');
ntrain = fread(fp1_1, 1, 'int32', 0, 'ieee-be');
rows = fread(fp1_1, 1, 'int32', 0, 'ieee-be');
cols = fread(fp1_1, 1, 'int32', 0, 'ieee-be');
train_data=zeros(ntrain, rows*cols);
for i = 1:ntrain
     train_data(i, :)=fread(fp1_1,(rows*cols), 'uint8');
end

%train label
filename1_2 = '/Users/Winnie/Downloads/三上/圖形識別/matlabprac/train-labels.idx1-ubyte';
fp1_2=fopen(filename1_2,'r');
magic = fread(fp1_2, 1, 'int32', 0, 'ieee-be');
ntrain = fread(fp1_2, 1, 'int32', 0, 'ieee-be');
train_tar=zeros(ntrain, 1);
for i = 1:ntrain
     train_tar(i, :)=fread(fp1_2,1, 'uint8');
end

%load test data
filename2_1 = '/Users/Winnie/Downloads/三上/圖形識別/matlabprac/t10k-images.idx3-ubyte';
fp2_1=fopen(filename2_1,'r');
magic = fread(fp2_1, 1, 'int32', 0, 'ieee-be');
ntest = fread(fp2_1, 1, 'int32', 0, 'ieee-be');
rows = fread(fp2_1, 1, 'int32', 0, 'ieee-be');
cols = fread(fp2_1, 1, 'int32', 0, 'ieee-be');
test=zeros(ntest, rows*cols);
for i = 1:ntest
     test(i, :)=fread(fp2_1,(rows*cols), 'uint8');
end

%test label
filename2_2 = '/Users/Winnie/Downloads/三上/圖形識別/matlabprac/t10k-labels.idx1-ubyte';
fp2_2=fopen(filename2_2,'r');
magic = fread(fp2_2, 1, 'int32', 0, 'ieee-be');
ntest = fread(fp2_2, 1, 'int32', 0, 'ieee-be');
test_tar=zeros(ntest, 1);
for i = 1:ntest
     test_tar(i, :)=fread(fp2_2,1, 'uint8');
end

trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
hiddenLayerSize = 32;
net = patternnet(hiddenLayerSize, trainFcn);
net.trainParam.lr=0.01;
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;
tar=zeros(ntrain, 10);
for i=1:ntrain
    tar(i,1+train_tar(i, :))=1;
end

[net,tr] = train(net, train_data', tar');
view(net);
figure, plotperform(tr);

figure;
hold on;
target=zeros(100, 10);
for i=1:100
    target(i,1+test_tar(i, :))=1;
end
for i = 1:10
    for j=1:10
      subplot(10, 10, (i-1)*10+j);
      img=reshape(test((i-1)*10+j,:), rows, cols)';
      actualLabel = test_tar((i-1)*10+j, :);
      pred((i-1)*10+j,:)=sim(net,test((i-1)*10+j,:)');
      [a, index]=max(pred((i-1)*10+j,:));
      index=index-1;
      imshow(img);
      title(actualLabel);
      text(0, 35, sprintf('predicted: %d',index));
    end
end

figure;
plotconfusion(target', pred');
