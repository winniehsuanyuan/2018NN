%two spiral
clear;
clc
qi=0:96; 
r=6.5.*(104-qi)./104; %radius
q=pi/16.*qi;          %degree
x1 = r.*sin(q);
y1 = r.*cos(q);
x2 = -r.*sin(q);
y2 = -r.*cos(q);

%add desired output to data
data=[x1' y1' (x1.*x1)' (y1.*y1)' (x1.*y1)' ones(97,1) zeros(97,1);
     x2' y2' (x2.*x2)' (y2.*y2)' (x2.*y2)' zeros(97,1) ones(97,1);];  
r=randperm(97*2);
data=data(r, :);

%net
I=5+1;  %input+const
J=100+1;  %hidden+const
K=2;    %class
n=97*2;

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
lowlimit=0.02;
itermax=25000;
iter=0;    %iteration
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
        oi=[data(i,1:5) 1]';
        dk=[data(i,6:7)]';
        for j=1:J-1
            sj(j)=wji(j, :)*oi;
            oj(j)=1/(1+exp(-sj(j)));   %sigmoidal
        end
        
        oj(J)=1.0;   %const
        
        for k=1:K
            sk(k)=wkj(k, :)*oj;
            ok(k)=1/(1+exp(-sk(k)));   %sigmoidal
        end
        error=error+sum(abs(dk-ok));
    %backward learning
        for k=1:K
        deltak(k)=(dk(k)-ok(k))*ok(k)*(1.0-ok(k));   %ok'(k)=ok(k)(1-ok(k))
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
axis square;
%plot the decision regions
for ix=1:1:161
    for iy=1:1:161
        dx=-8+0.1*(ix-1);
        dy=-8+0.1*(iy-1);
        oi=[dx dy dx*dx dy*dy dx*dy 1]';
        
        for j=1:J-1
            sj(j)=wji(j,:)*oi;
            oj(j)=1/(1+exp(-sj(j)));   %sigmoidal

        end
        oj(J)=1.0;
        
        for k=1:K
            sk(k)=wkj(k,:)*oj;
            ok(k)=1/(1+exp(-sk(k)));
        end

        if ok(1, 1)> 0.5                %sigmoidal
            plot(dx, dy, '.', 'color', [0.6, 0.2, 0.2]);
        else
            plot(dx, dy, '.', 'color', [0.2, 0.2, 0.6]);
        end
    end
end

plot(x1, y1, 'ro');
plot(x2, y2, 'b+');

