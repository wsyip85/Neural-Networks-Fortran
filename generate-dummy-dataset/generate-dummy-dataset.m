clc; close all; clear all;

pkg load communications

x1 = linspace( 0, 10, 100 );

x2 = linspace( 0, 10, 100 );

seed = 9012;

[X1, X2] = meshgrid(x1,x2);

X = [ X1(:), X2(:) ];

idx = randint( length(X), 1, [1 length(X)], seed );

X = X(idx,:);

Y = X(:,1).^2 + X(:,2).^2;

nfold = 10;

ntest = length(Y) * nfold / 100;

ntrain = length(Y) * (100-nfold) / 100;

M = [X Y];

Mtest01 = M(ntest*0+1:ntest*1,:); Mtrain01 = M([          ntest*1+1:length(M)],:);
Mtest02 = M(ntest*1+1:ntest*2,:); Mtrain02 = M([1:ntest*1,ntest*2+1:length(M)],:);
Mtest03 = M(ntest*2+1:ntest*3,:); Mtrain03 = M([1:ntest*2,ntest*3+1:length(M)],:);
Mtest04 = M(ntest*3+1:ntest*4,:); Mtrain04 = M([1:ntest*3,ntest*4+1:length(M)],:);
Mtest05 = M(ntest*4+1:ntest*5,:); Mtrain05 = M([1:ntest*4,ntest*5+1:length(M)],:);
Mtest06 = M(ntest*5+1:ntest*6,:); Mtrain06 = M([1:ntest*5,ntest*6+1:length(M)],:);
Mtest07 = M(ntest*6+1:ntest*7,:); Mtrain07 = M([1:ntest*6,ntest*7+1:length(M)],:);
Mtest08 = M(ntest*7+1:ntest*8,:); Mtrain08 = M([1:ntest*7,ntest*8+1:length(M)],:);
Mtest09 = M(ntest*8+1:ntest*9,:); Mtrain09 = M([1:ntest*8,ntest*9+1:length(M)],:);
Mtest10 = M(ntest*9+1:ntest*10,:); Mtrain10 = M([1:ntest*9,ntest*10+1:length(M)],:);

dlmwrite('../../data/test01.txt',Mtest01); dlmwrite('../../data/train01.txt',Mtrain01);
dlmwrite('../../data/test02.txt',Mtest02); dlmwrite('../../data/train02.txt',Mtrain02);
dlmwrite('../../data/test03.txt',Mtest03); dlmwrite('../../data/train03.txt',Mtrain03);
dlmwrite('../../data/test04.txt',Mtest04); dlmwrite('../../data/train04.txt',Mtrain04);
dlmwrite('../../data/test05.txt',Mtest05); dlmwrite('../../data/train05.txt',Mtrain05);
dlmwrite('../../data/test06.txt',Mtest06); dlmwrite('../../data/train06.txt',Mtrain06);
dlmwrite('../../data/test07.txt',Mtest07); dlmwrite('../../data/train07.txt',Mtrain07);
dlmwrite('../../data/test08.txt',Mtest08); dlmwrite('../../data/train08.txt',Mtrain08);
dlmwrite('../../data/test09.txt',Mtest09); dlmwrite('../../data/train09.txt',Mtrain09);
dlmwrite('../../data/test10.txt',Mtest10); dlmwrite('../../data/train10.txt',Mtrain10);

