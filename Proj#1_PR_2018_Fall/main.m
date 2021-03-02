%project1_0516218
clear;
clc
%1
m=3;
a=2;
x=linspace(m-3*a, m+3*a);
g=1/(a*((2*pi)^(1/2)))*exp((-1/2)*((x-m)/a).^2);
subplot(6, 4, 1);
plot(x, g);
title('1-d Gaussian function');

%2
mx=1;
my=2;
ax=1;
ay=1;
X=linspace(mx-3*ax, mx+3*ax);
Y=linspace(my-3*ay, my+3*ay);
[x,y] = meshgrid(X,Y);
g=1/(2*pi*ax*ay)*exp(-1/2*((x-mx).^2/ax^2+(y-my).^2/ay^2));
subplot(6, 4, 5);
surf(x,y,g);
title('2-d Gaussian function');

%3
m=0;
a=1;
x=a.*randn(100)+m;
subplot(6, 4, 9);
histogram(x);
title('1-d histogram for 1-d Gaussian random data');

%4&5
m=[1 2];
a=[3 0;0 4];
x=mvnrnd(m, a, 10000);
counts=hist3(x, 'Ctrs',{-6:0.3:6 -4:0.3:8}, 'CDataMode','auto','FaceColor','interp');

subplot(6, 4, 13);
plot(x(:,1), x(:,2), '.');
title('2-d Gaussian random data');

subplot(6, 4, 17);
hist3(x, 'Ctrs',{-6:0.3:6 -4:0.3:8}, 'CDataMode','auto','FaceColor','interp');
title('2-d histogram for 2-d Gaussian random data');

subplot(6, 4, 21);
contour(-6:0.3:6, -4:0.3:8, counts, 'ShowText', 'on');
title('contour for the previous 2-d histogram');

%6
x=linspace(-6, 6);
y=x-1;
subplot(6, 4, 2);
plot(x, y);
axis tight;
title('x-y-1=0');

%7
subplot(6, 4, 6);
ezplot('x^2+y^2=1', [-1.5, 1.5]);
axis equal;
title('x^2+y^2-1=0');

%8
subplot(6, 4, 10);
ezplot('x^2+y^2/4=1', [-3, 3]);
axis equal;
title('x^2+y^2/4=1');

%9
subplot(6, 4, 14);
ezplot('x^2-y^2/4=1');
axis equal;
title('x^2-y^2/4=1');

%10
x=linspace(0,100, 100);
y=2*x;
subplot(6, 4, 18);
plot(x, y, '.');
title('2x-y=0');

%11
x=linspace(-2,2 , 50);
y1=sqrt(4-x.^2);
y2=-sqrt(4-x.^2);
subplot(6, 4, 22);
hold on;
box on;
plot(x, y1, 'b.');
plot(x, y2, 'b.');
axis equal;
title('x^2+y^2=4');
hold off;

%12
x=linspace(-2,2 , 50);
y1=sqrt(1-x.^2/4);
y2=-sqrt(1-x.^2/4);
subplot(6, 4, 3);
hold on;
box on;
plot(x, y1, 'b.');
plot(x, y2, 'b.');
axis equal;
title('x^2/4+y^2=1');
hold off;

%13
x1=linspace(-1,-0.1, 50);
x2=linspace(0.1,1, 50);
y1=1./x1;
y2=1./x2;
subplot(6, 4, 7);
hold on;
box on;
plot(x1, y1, 'b.');
plot(x2, y2, 'b.');
axis([-1 1 -10 10]);
title('xy=1');
hold off;

%14
i=0:96; 
r=6.5.*(104-i)./104; %radius
q=pi/16.*i;          %degree
x = r.*sin(q);
y = r.*cos(q);
subplot(6, 4, 11);
plot(x, y, 'ro');
axis equal 
hold on;
x = -r.*sin(q);
y = -r.*cos(q);
plot(x, y, 'b+');
title('two spirals');


%15
x=linspace(0, 1);
y=linspace(0, 1);
z = @(x,y)(x-y+0.5); 
subplot(6, 4, 15);
s=fsurf(z, [0, 1], 'y', 'edgecolor', 'y');
alpha(0.3);
axis square;
xlabel('x');
ylabel('y');
zlabel('z');
title('plane z=x-y+0.5 that separate 2 classes');
box on;
hold on;
plot3(0, 1, 1, 'bo', 'MarkerSize', 10);
plot3(0, 0, 0, 'rx', 'MarkerSize', 10);
plot3(1, 0, 0, 'rx', 'MarkerSize', 10);
plot3(1, 0, 1, 'rx', 'MarkerSize', 10);
plot3(0, 0, 1, 'bo', 'MarkerSize', 10);
plot3(1, 1, 0, 'rx', 'MarkerSize', 10);
plot3(1, 1, 1, 'bo', 'MarkerSize', 10);
plot3(0, 1, 0, 'bo', 'MarkerSize', 10);
hold off;

%16
w=6;
r=10;
d=-2;
ts=200;
ts1=10*ts;
done=0; 
tmp1=[];

while ~done
    tmp=[2*(r+w/2)*(rand(ts1,1)-0.5) (r+w/2)*rand(ts1,1)]; 
    tmp(:,3)=sqrt(tmp(:,1).*tmp(:,1)+tmp(:,2).*tmp(:,2)); 
    idx=find([tmp(:,3)>r-w/2] & [tmp(:,3)<r+w/2]);
    tmp1=[tmp1;tmp(idx,1:2)];
    if length(idx)>= ts
        done=1;
    end
end

data=[tmp1(1:ts,:) zeros(ts,1);
    [tmp1(1:ts,1)+r -tmp1(1:ts,2)-d ones(ts,1)]];
subplot(6, 4, 19);
plot(data(1:ts,1),data(1:ts,2),'bo',data(ts+1:end,1),data(ts+1:end,2),'rs');
title(['double moon problem']),
axis([-r-w/2 2*r+w/2 -r-w/2-d r+w/2]);

%17
x = 0:pi/50:2*pi;
t0=2*pi;
subplot(6, 4, 4);
h1=plot(x,sin(x*2*pi/(t0)),'linewidth',3);
title('sine function with period T0');
axis([-inf inf -1 1])

subplot(6, 4, 8);
h2=plot(x,sin(x*2*pi/(t0/2)),'linewidth',3); 
title('sine function with period T0/2');
axis([-inf inf -1 1])

subplot(6, 4, 12);
h3=plot(x,sin(x*2*pi/(t0/3)),'linewidth',3);
title('sine function with period T0/3');
axis([-inf inf -1 1])

subplot(6, 4, 16);
h4=plot(x,sin(x*2*pi/(t0/4)),'linewidth',3);
title('sine function with period T0/4');
axis([-inf inf -1 1])

subplot(6, 4, 20);
h5=plot(x,sin(x*2*pi/(t0/5)),'linewidth',3); 
title('sine function with period T0/5');
axis([-inf inf -1 1])

dx=5*pi/360;

while x<6*pi 
x=x+dx;
set(h1,'xdata',x,'ydata',sin(x*2*pi/(t0))); 
set(h2,'xdata',x,'ydata',sin(x*2*pi/(t0/2))); 
set(h3,'xdata',x,'ydata',sin(x*2*pi/(t0/3))); 
set(h4,'xdata',x,'ydata',sin(x*2*pi/(t0/4))); 
set(h5,'xdata',x,'ydata',sin(x*2*pi/(t0/5))); 
drawnow
end

%20
filename = '/Users/Winnie/Downloads/三上/圖形識別/hw/hw1/Proj#1_軒轅照雯_PR_2018_Fall/t10k-images.idx3-ubyte';
fp=fopen(filename,'r');
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
rows = fread(fp, 1, 'int32', 0, 'ieee-be');
cols = fread(fp, 1, 'int32', 0, 'ieee-be');
test_x = zeros(150,rows*cols);
for i = 1:150
            temp = fread(fp,(rows*cols), 'uchar');
            test_x(i,:) = temp;
end
test=reshape(test_x, 15*rows, 10*cols);
subplot(6, 4, 23);
imshow(test);
title('150 patterns of MNIST database');


