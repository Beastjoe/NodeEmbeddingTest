% FastMMD
% Ji Zhao@CMU
% zhaoji84@gmail.com
% 10/26/2013

clear;

allSgm = 10.^(-2:0.2:2);
allSgm = allSgm(:);
nBasis = 2^10;

%% generate data
[X, Y] = GenSamp(1); % 1 -- is_rand
xPos = X(Y == 1, :);
xNeg = X(Y == -1, :);

%idxA = randsample(1:512, 100);

fileA = fopen('../../embedding/node2vec/output/kronecker-2_1_2d.emb','r');
A = fscanf(fileA, '%d %f %f', [3 512]).';
A(:,1) = [];
A(101:512,:) = [];
%A = A(1:500,:);
fileB = fopen('../../embedding/node2vec/output/kronecker-2_0_2d.emb','r');
B = fscanf(fileB, '%d %f %f', [3 512]).';
B(:,1) = [];
B(101:512,:) = [];

fprintf(1, '-------------beginning-------------\n')
%% MMD
%tic, d1 = MMD(xPos, xNeg, allSgm, 'biased'); f1 = MMD(xPos, xNeg, allSgm, 'unbiased'); toc
tic, [d1, d2, ds1, ds2, ds3] = MMD3(A, B, allSgm); toc

%% FastMMD via Fastfood
tic, [d1, d2] = MMDFastfood(A, B, allSgm, nBasis); toc

%% MMD-linear
tic, f5 = MMDlinear(A, B, allSgm); toc
