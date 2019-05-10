Im = imread("./image_2/000099_10.png");
% imshow(I);
I = double(Im);
% three channels colored image
r = I(:,:,1); g = I(:,:,2); b = I(:,:,3);
I = [r(:),g(:),b(:)];
i = 0;
j = 0;
center = [0 0 0];
while(true)
    s(1) = mean(I(:,1));
    s(2) = mean(I(:,2));
    s(3) = mean(I(:,3));
    i = i+1;
    while(true)
        j = j+1;
        seedvec = repmat(s,[size(I,1),1]);
        dist = sum((sqrt((I-seedvec).^2)),2);
        distth = 0.25*max(dist);
        qualified = dist<distth;
        nr = I(:,1);
        ng = I(:,2);
        nb = I(:,3);
        ns(1) = mean(nr(qualified));
        ns(2) = mean(ng(qualified));
        ns(3) = mean(nb(qualified));
        if isnan(ns)
            break;
        end
        if (s == ns) | j>10
            j=0;
            I(qualified,:) = [];
            center(i,:) = ns;
            break;
        end
        s = ns;
    end
    if isempty(I) || i>10
        i = 0;
        break;
    end
    
end
centers = sqrt(sum((center.^2),2));
[centers,idx]= sort(centers);
while(true)
    newcenter = diff(centers);
    intercluster =25;
    a = (newcenter<=intercluster);
    centers(a,:) = [];
    idx(a,:)=[];
    if nnz(a)==0
        break;
    end
end
centertemp = center;
center =centertemp(idx,:);
% [~,idxsort] = sort(centers) ;
vecr = repmat(r(:),[1,size(center,1)]);
vecg = repmat(g(:),[1,size(center,1)]);
vecb = repmat(b(:),[1,size(center,1)]);
distr = (vecr - repmat(center(:,1)',[numel(r),1])).^2;
distgr = (vecg - repmat(center(:,2)',[numel(r),1])).^2;
distb = (vecb - repmat(center(:,3)',[numel(r),1])).^2;
distance = sqrt(distr+distgr+distb);
[~,label_vector] = min(distance,[],2);
L = reshape(label_vector,size(r));
% overlay segmentation results with original
% images to visualize the segmentation quality
% for comparison:
% L = clustering(I,5,'imsegkmeans');
res = labeloverlay(Im,L);
figure, imshow(res);
% write result
imwrite(L,'seg000001.png');
ground_truth = imread("./semantic/000099_10.png");
L = cast(L,'uint8');
immse(ground_truth,L)