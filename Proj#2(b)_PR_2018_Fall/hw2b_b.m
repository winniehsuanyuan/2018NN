%gaussion relu
%mnist 0~9
clear all;
clc

%load train data
filename1_1 = '/Users/Winnie/Downloads/三上/圖形識別/matlabprac/train-images.idx3-ubyte';
fp1_1=fopen(filename1_1,'r');
magic = fread(fp1_1, 1, 'int32', 0, 'ieee-be');
ntrain = fread(fp1_1, 1, 'int32', 0, 'ieee-be');
rows = fread(fp1_1, 1, 'int32', 0, 'ieee-be');
cols = fread(fp1_1, 1, 'int32', 0, 'ieee-be');
train=zeros(ntrain, rows*cols);
for i = 1:ntrain
     train(i, :)=fread(fp1_1,(rows*cols), 'uint8');
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


%data
train=normalize(train,'range');%normalize
data=zeros(ntrain, rows*cols+10);
for i=1:ntrain
    data(i,1:rows*cols)=train(i, :);
    data(i,rows*cols+1+train_tar(i, :))=1;
end

%train net
I=rows*cols+1;  %input+const
J=250+1;  %hidden+const
K=10;    %class
n=ntrain;

%initialize
wkj=randn(K, J);
wkj_temp=zeros(size(wkj));
wji=randn(J-1, I);
old_dwkj=zeros(size(wkj));
old_dwji=zeros(size(wji));
oi=[zeros(I-1, 1);1];
sj=zeros(J-1,1);
oj=[sj;1];
sk=zeros(K,1);
ok=zeros(K,1);
dk=zeros(K,1);
lowlimit=0.1;
itermax=100;
iter=0;   
%eta=0.1;beta=0.9;   %add momentum term
eta=0.01;beta=0.0;  

erroravg=100;
besterror=100;

%internal variables
deltak=zeros(1, K);
sumback=zeros(1, J-1);
deltaj=zeros(1, J-1);

%for a=0:5
%    wkj=randn(K, J)+a;
%    wji=randn(J-1, I)+a;
%    iter=0;
tic
while (erroravg>lowlimit) && (iter<itermax)
    iter=iter+1;
    error=0;
    %forward computation
    for i=1:n
        oi=[data(i,1:rows*cols) 1]';
        dk=[data(i,rows*cols+1:rows*cols+10)]';
        for j=1:J-1
            sj(j)=wji(j, :)*oi;         
            oj(j)=max(0, sj(j));      %ReLu
        end
        oj(J)=1.0;   %const
        
        for k=1:K
            sk(k)=wkj(k, :)*oj;
            ok(k)=max(0, sk(k));
        end
        error=error+sum(abs(dk-ok));
    %backward learning
        for k=1:K
           deltak(k)=(dk(k)-ok(k))*(sk(k)>0);   %ok'(k)=0, 1;
        end
    
        for j=1:J
            for k=1:K
                wkj_temp(k,j)=wkj(k,j)+eta*deltak(k)*oj(j);
            end
        end
    
        for j=1:J-1
            sumback(j)=0.0;
            for k=1:K
                sumback(j)=sumback(j)+deltak(k)*wkj(k,j);
            end
            deltaj(j)=sumback(j)*(sj(j)>0);   %oj'(j)=0,1
        end
    
        for i=1:I
            for j=1:J-1
                wji(j,i)=wji(j,i)+eta*deltaj(j)*oi(i);
            end
        end
        wkj=wkj_temp;
    end
    
    ite(iter)=iter;     
    erroravg=error/n;
    error_r(iter)=erroravg;
end
toc

 figure;
 hold on;
 plot(ite, error_r);
 xlabel('iterations');
 ylabel('average error');
% if erroravg<besterror
%    bestwkj=wkj;
%    bestwji=wji;
%    besterror=erroravg;
%    besta=a;
% end
%end


figure;
hold on;
%test
test=normalize(test,'range');
data_=zeros(100, rows*cols+10);
test_oks=zeros(10, 100);
test_dks=zeros(10, 100);

for i=1:100
    data_(i,1:rows*cols)=test(i, :);
    data_(i,rows*cols+1+test_tar(i, :))=1;
    test_dks(test_tar(i, 1)+1,i)=1;
end

predict=zeros(100, 1);
for i=1:100
    oi=[data_(i,1:rows*cols) 1]';
    dk=[data_(i,rows*cols+1:rows*cols+10)]';
    for j=1:J-1
        sj(j)=wji(j, :)*oi;
        oj(j)=max(0, sj(j));      %ReLu
    end
    oj(J)=1.0;
    for k=1:K
        sk(k)=wkj(k, :)*oj;
        ok(k)=max(0, sk(k));      %ReLu
    end
    [m,index]=max(ok);
    test_oks(index, i)=1;
    predict(i,:)=index-1;
    subplot(10, 10, i);
    temp=test(i,:);
    temp=reshape(temp, rows, cols)';
    imshow(temp);
    title([test_tar(i, :)]);
    text(0, 35, sprintf('predicted: %d',predict(i,:)));
end
figure;
plotconfusion(test_dks,test_oks);

   

   
