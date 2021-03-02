%double moon
clear all;
clc
N=250;
theta1 = linspace(-180,180, N)*pi/360;
r=8;
x1 = -5 + r*sin(theta1)+randn(1,N); 
y1 = r*cos(theta1)+randn(1,N);
x2 = 5 + r*sin(theta1)+randn(1,N);
y2 = -r*cos(theta1)+randn(1,N);

%add desired output to data
data=[x1.', y1.', ones(250, 1), zeros(250, 1); 
      x2.', y2.', zeros(250, 1), ones(250, 1)];   
r=randperm(500);
data=data(r, :);               

%net
I=2+1;  %input+const
J=3+1;  %hidden+const
K=2;    %class
n=500;

%initialize
wkj=randn(K, J);
wkj_temp=zeros(size(wkj));
wji=randn(J-1, I);
old_dwkj=zeros(size(wkj));
old_dwji=zeros(size(wji));
oi=[0 0 1]';
sj=[0 0 0]';
oj=[0 0 0 1]';
sk=[0 0]';
ok=[0 0]';
dk=[0 0]';
lowlimit=0.002;
itermax=5000;
iter=0;   
%eta=0.7;beta=0.3;  %add momentum term
eta=0.01;beta=0.0;  

erroravg=10;

%internal variables
deltak=zeros(1, K);
sumback=zeros(1, J-1);
deltaj=zeros(1, J-1);

while (erroravg>lowlimit) && (iter<itermax)
    iter=iter+1;
    error=0;
    %forward computation
    for i=1:n
        oi=[data(i,1:2) 1]';
        dk=[data(i,3:4)]';
        for j=1:J-1
            sj(j)=wji(j, :)*oi;
            oj(j)=1/(1+exp(-sj(j)));
        end
        oj(J)=1.0;  
        
        for k=1:K
            sk(k)=wkj(k, :)*oj;
            ok(k)=1/(1+exp(-sk(k)));
        end
        
        error=error+sum(abs(dk-ok));
    %backward learning
        for k=1:K
        deltak(k)=(dk(k)-ok(k))*ok(k)*(1.0-ok(k));
        end
    
        for j=1:J
            for k=1:K
                wkj_temp(k,j)=wkj(k,j)+eta*deltak(k)*oj(j)+beta*old_dwkj(k,j);
                old_dwkj(k, j)=eta*deltak(k)*oj(j)+beta*old_dwkj(k,j);
            end
        end
    
        for j=1:J-1
            sumback(j)=0.0;
            for k=1:K
                sumback(j)=sumback(j)+deltak(k)*wkj(k,j);
            end
            deltaj(j)=oj(j)*(1.0-oj(j))*sumback(j);
        end
    
        for i=1:I
            for j=1:J-1
                wji(j,i)=wji(j,i)+eta*deltaj(j)*oi(i)+beta*old_dwji(j,i);
                old_dwji(j,i)=eta*deltaj(j)*oi(i)+beta*old_dwji(j,i);
            end
        end
        wkj=wkj_temp;
    
    end
    
    ite(iter)=iter;     
    erroravg=error/n;
    error_r(iter)=erroravg; 
end

figure;
hold on;
plot(ite, error_r);
xlabel('iterations');
ylabel('average error');

figure;
hold on;
axis([-15 15 -15 15]);
%plot the decision regions
for ix=1:1:301
    for iy=1:1:301
        dx=-15+0.1*(ix-1);
        dy=-15+0.1*(iy-1);
        oi=[dx dy 1]';
        
        for j=1:J-1
            sj(j)=wji(j,:)*oi;
            oj(j)=1/(1+exp(-sj(j)));
        end
        oj(J)=1.0;
        
        for k=1:K
            sk(k)=wkj(k,:)*oj;
            ok(k)=1/(1+exp(-sk(k)));
        end

        if ok(1, 1)> 0.5
            plot(dx, dy, '.', 'color', [0.2, 0.4, 0.6]);
        else
            plot(dx, dy, '.', 'color', [0.6, 0.4, 0.2]);
            end

        end
end

plot(x1,y1,'bo');
plot(x2,y2,'rs');
