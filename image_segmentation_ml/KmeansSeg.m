Im = imread("./image_2/000099_10.png");
refer = histeq(Im);
imshow(refer);
% number of clusters, parameter.
k = 8;
NormalizeInput = true;
NumAttempt = 3;
MaxIterations = 100;
Threshold = 0.0001;
% casting input to single datatype
I = single(Im);
[m,n,c] = size(I);
X = reshape(I,m*n,[]);
% normalize input
avgChannel = mean(X);
stdDevChannel = std(X);
zeroLoc = stdDevChannel==0;
stdDevChannel(zeroLoc) = 1;
temp = (X - avgChannel)./stdDevChannel;
X = temp;
%kd based KNN search algorithm to speed up
[L,NormCen] = images.internal.ocvkmeans(X,k,NumAttempt,MaxIterations,Threshold);
%denormalization
Centers = NormCen .* stdDevChannel + avgChannel;
Centers = cast(Centers,'uint8');
L = reshape(L,m,n);
L = cast(L,'uint8');
res = labeloverlay(Im,L);
figure, imshow(res);
ground_truth = imread("./semantic/000099_10.png");
immse(ground_truth,L)