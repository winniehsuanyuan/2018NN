%gaussian matlab toolbox
clear all;
clc
m1=[0 0];
a1=[1 0;0 1];
m2=[14 0];
a2=[1 0;0 4];
m3=[7 14];
a3=[4 0;0 1];
m4=[7 7];
a4=[1 0;0 1];
rng default  
c1 = mvnrnd(m1,a1,150);
c2 = mvnrnd(m2,a2,150);
c3 = mvnrnd(m3,a3,150);
c4 = mvnrnd(m4,a4,150);

input=[c1;c2;c3;c4];
target=[ones(150, 1) zeros(150, 3);
        zeros(150, 1) ones(150, 1) zeros(150, 2);
        zeros(150, 2) ones(150, 1) zeros(150, 1);
        zeros(150, 3) ones(150, 1)];
x = input';
t = target';

trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
hiddenLayerSize = 3;
net = patternnet(hiddenLayerSize, trainFcn);
net.trainParam.lr=0.01;
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 20/100;

% Train the Network
[net,tr] = train(net,x,t);
% Test the Network
y = net(x);
e = gsubtract(t,y);

view(net);
figure, plotperform(tr);
figure, plottrainstate(tr);
figure, ploterrhist(e);
figure, plotconfusion(t,y);
figure, plotroc(t,y);

figure;
hold on;
%plot the decision regions
for ix=1:1:101
    for iy=1:1:101
        dx=-5+0.25*(ix-1);
        dy=-5+0.25*(iy-1);
        pred=sim(net,[dx;dy]); 
       
        if pred(1, 1)==1
            plot(dx, dy, '.', 'color', [0.6, 0.2, 0.2]);
        elseif pred(2, 1)==1
            plot(dx, dy, '.', 'color', [0.2, 0.6, 0.2]);
        elseif pred(3, 1)==1
            plot(dx, dy, '.', 'color', [0.2, 0.2, 0.6]);
        elseif pred(4, 1)==1
            plot(dx, dy, '.', 'color', [0.6, 0.6, 0.2]);
        else
            [m,index]=max(pred);
            switch index
              case 1
                 plot(dx, dy, '.', 'color', [0.6, 0.2, 0.2]);
              case 2
                 plot(dx, dy, '.', 'color', [0.2, 0.6, 0.2]);
              case 3
                 plot(dx, dy, '.', 'color', [0.2, 0.2, 0.6]);
              case 4
                 plot(dx, dy, '.', 'color', [0.6, 0.6, 0.2]);
            end
        end
    end
end

%plot original data classes
plot(c1(:,1),c1(:,2),'r.');
plot(c2(:,1),c2(:,2),'g.');
plot(c3(:,1),c3(:,2),'b.');
plot(c4(:,1),c4(:,2),'y.');        
   
