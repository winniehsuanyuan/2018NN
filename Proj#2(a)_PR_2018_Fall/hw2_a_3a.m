%gaussion sigmoidal
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

%add desired output to data
data=[c1 ones(150, 1) zeros(150, 3);
      c2 zeros(150, 1) ones(150, 1) zeros(150, 2);
      c3 zeros(150, 2) ones(150, 1) zeros(150, 1);
      c4 zeros(150, 3) ones(150, 1)];  
r=randperm(150*4);
data=data(r, :);

%net
I=2+1;  %input+const
J=3+1;  %hidden+const
K=4;    %class
n=600;

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
lowlimit=0.002;
itermax=15000;
iter=0;   
%eta=0.7;beta=0.3;   %add momentum term
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
        dk=[data(i,3:6)]';
        for j=1:J-1
            sj(j)=wji(j, :)*oi;
            oj(j)=1/(1+exp(-sj(j)));    %sigmoidal
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
%plot the decision regions
for ix=1:1:251
    for iy=1:1:251
        dx=-5+0.1*(ix-1);
        dy=-5+0.1*(iy-1);
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
            plot(dx, dy, '.', 'color', [0.6, 0.2, 0.2]);
        elseif ok(2,1) > 0.5
            plot(dx, dy, '.', 'color', [0.2, 0.6, 0.2]);
        elseif ok(3,1) > 0.5
            plot(dx, dy, '.', 'color', [0.2, 0.2, 0.6]);
        elseif ok(4,1) > 0.5
            plot(dx, dy, '.', 'color', [0.6, 0.6, 0.2]);
        else
            [m,index]=max(ok);
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



