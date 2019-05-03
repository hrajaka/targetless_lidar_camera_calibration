% read input pcl files
pcl = pcread('./training/000001.pcd');
% pcshow(pcl);
% key parameter of distance
distance = 0.5;
labelsize = [size(pcl.Location, 1) 1];
[pc,valid] = removeInvalidPoints(pcl);
count = pc.Count;
L = zeros(count,1,'uint32');
newLabel = 0;
for i = 1:count
    if L(i) ~= 0
        continue;
    end
    %KNN K-d based search algorithm
    index = findNeighborsInRadius(pc,pc.Location(i,:),distance);
    for k = 1:numel(index)
        j = index(k);
        if L(j) > 0 && L(i) > 0
            if L(j) > L(i)
                L(L==L(j)) = L(i);
            elseif L(j) < L(i)
                L(L==L(i)) = L(j);
            end
        else
            if L(j) > 0
                L(i) = L(j);
            elseif L(i) > 0
                L(j) = L(i);
            end
        end
    end
        if L(i) == 0
        newLabel = newLabel+1;
        L(index) = newLabel;
    end
end
uniqueLabels = unique(L);
numClusters = cast(length(uniqueLabels), 'like', pcl.Location);
for k = 1:numClusters
    L(L == uniqueLabels(k)) = k;
end
labels = zeros(labelsize,'uint32');
labels(valid) = L;
pcshow(pcl.Location,labels);
colormap(hsv(numClusters));
%write results to pcd file
fileID = fopen('seg000001.pcd','w');
fwrite(fileID,labels,'uint32');