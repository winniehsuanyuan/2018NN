clear;
clc
%toolbox
f=0.1*ones(4, 4);
g=0.1*ones(4, 4);

conv2(f, g)

%self
a=[zeros(3, 10);
   zeros(4, 3) f zeros(4, 3);
   zeros(3, 10);];
b=g;
c=zeros(7, 7);
for i=1:7
    for j=1:7
        for x=1:4
            for y=1:4
                c(i, j)=c(i, j)+a(i+x-1, j+y-1)*b(x, y);
            end
        end
    end
end

display(c);

