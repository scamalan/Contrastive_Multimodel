% %create the label and label-image file for 32X32 images
f_name = 'C:\Users\camalas.DEACNET\Project\Planet_S1S2_training_data';
S1 = imread(strcat(f_name,'\','s1_test_20210808_3173x3224.tif'));
S2 = imread(strcat(f_name,'\','s2_test_20210807_3173x3224.tif'));
P = imread(strcat(f_name,'\','planet_20210807\4772862_1933110_2021-08-07_2455_BGRN_Analytic_toar_clip.tif'));

% path='C:\Users\camalas.DEACNET\Project\Analyze_ponds\Pond_Images\For_32\';
path = 'C:\Users\camalas.DEACNET\Project\Planet_S1S2_training_data\';
folder = dir(strcat(path,'**\*_32.tif'));
dest = strcat(path,'tiled_images_v2\');
fld_mask = 'Masks\';
fld_s2 = 'S2s\';
fld_p = 'Ps\';
data_1_s2 = [];
data_2_s2 = [];
data_p = [];
label = [];
list = [];
% list = load(strcat(dest,'tile_list_32899.mat'));
% list=list.list;
% data_s2 = load(strcat(dest,'data_s2_32899.mat'));
% data_s2=data_s2.data_s2;
% data_p = load(strcat(dest,'data_p_32899.mat'));
% data_p=data_p.data_p;
% label = load(strcat(dest,'label_32899.mat'));
% label=label.label;
count =0;
rgbsw_1 = S2(:,:,[4,3,2,11]);
rgbsw_2 = S2(:,:,[4,3,2,12]); %it was 11 before
rgbnr = P;
tile_sz =32;
msk = imread(strcat(path,'Thr_Labels_final\s2_test_20210807_3173x3224_correct.png'));
[row,col] = size(msk);
rr = mod(row,tile_sz);
cr = mod(col,tile_sz);
        
% num_sx = floor(sz(1)/32);
% num_sy = floor(sz(2)/32);

% msk = imread(strcat(path,'Thr_Labels_final\s2_test_20210807_3173x3224_correct.png'));
for j=1:(tile_sz/2):row-(rr+(tile_sz/2))
    for k=1:(tile_sz/2):col-(cr+(tile_sz/2))
        msk_ind = msk(((j-1)+1):(j-1)+tile_sz,((k-1)+1):(k-1)+tile_sz);
        rgbsw_ind_1 = uint8(rescale(rgbsw_1(((j-1)+1):(j-1)+tile_sz,((k-1)+1):(k-1)+tile_sz,:),0,255));
        rgbsw_ind_2 = uint8(rescale(rgbsw_2(((j-1)+1):(j-1)+tile_sz,((k-1)+1):(k-1)+tile_sz,:),0,255));
        rgbnr_ind = uint8(rescale(rgbnr(((j-1)+1):(j-1)+tile_sz,((k-1)+1):(k-1)+tile_sz,:),0,255));

        
        for i=1:8
            num_lbl(i)=sum(msk_ind(:)==(i-1));
        end
        
        count=count+1;
        
        [val,idx] = max(num_lbl);
        lbl = idx-1;
%         imwrite(msk_ind,strcat(dest,fld_mask,'mask_',string(j),'-',string(k),'_',string(count),'.tif'));
%         imwrite(rgbnr_ind,strcat(dest,fld_s2,'s2_',string(j),'-',string(k),'_',string(count),'.tif'));
%         imwrite(rgbsw_ind,strcat(dest,fld_p,'p_',string(j),'-',string(k),'_',string(count),'.tif'));
%         save(strcat(dest,fld_mask,'mask_',string(j),'-',string(k),'_',string(count),'.mat'),'msk_ind');
%         save(strcat(dest,fld_s2,'s2_',string(j),'-',string(k),'_',string(count),'.mat'),'rgbnr_ind');
%         save(strcat(dest,fld_p,'p_',string(j),'-',string(k),'_',string(count),'.mat'),'rgbsw_ind');
        data_ind_p = [reshape(rgbnr_ind(:,:,1)',1,[]),reshape(rgbnr_ind(:,:,2)',1,[]),reshape(rgbnr_ind(:,:,3)',1,[]),reshape(rgbnr_ind(:,:,4)',1,[])];
        data_ind_1_s2 = [reshape(rgbsw_ind_1(:,:,1)',1,[]),reshape(rgbsw_ind_1(:,:,2)',1,[]),reshape(rgbsw_ind_1(:,:,3)',1,[]),reshape(rgbnr_ind(:,:,4)',1,[])];
        data_ind_2_s2 = [reshape(rgbsw_ind_2(:,:,1)',1,[]),reshape(rgbsw_ind_2(:,:,2)',1,[]),reshape(rgbsw_ind_2(:,:,3)',1,[]),reshape(rgbnr_ind(:,:,4)',1,[])];

        data_1_s2 = [data_1_s2;data_ind_1_s2];        
        data_2_s2 = [data_2_s2;data_ind_2_s2];
        data_p = [data_p;data_ind_p];
        label = [label;lbl];
        list = [list;count,j,k];
        
    end
    
end

 save(strcat(dest,'label.mat'),'label');
 save(strcat(dest,'data_1_s2.mat'),'data_1_s2');
 save(strcat(dest,'data_2_s2.mat'),'data_2_s2');
 save(strcat(dest,'data_p.mat'),'data_p');
 save(strcat(dest,'tile_list.mat'),'list');

%% split the data into train, test, and validation
Label=label;
data_table = table ([Label],[data_1_s2]);
splitIndices = splitlabels(data_table,[0.7 0.2 0.1]);
countlabels(Label(splitIndices{3}))

train_data_1_s2 = data_1_s2(splitIndices{1},:);
train_data_2_s2 = data_2_s2(splitIndices{1},:);
train_data_p = data_p(splitIndices{1},:);
train_label = Label(splitIndices{1},:);
train_list = list(splitIndices{1},:);

test_data_1_s2 = data_1_s2(splitIndices{2},:);
test_data_2_s2 = data_2_s2(splitIndices{2},:);
test_data_p = data_p(splitIndices{2},:);
test_label = Label(splitIndices{2},:);
test_list = list(splitIndices{2},:);

val_data_1_s2 = data_1_s2(splitIndices{3},:);
val_data_2_s2 = data_2_s2(splitIndices{3},:);
val_data_p = data_p(splitIndices{3},:);
val_label = Label(splitIndices{3},:);
val_list = list(splitIndices{3},:);

save('train_data_1_s2.mat','train_data_1_s2');
save('train_data_2_s2.mat','train_data_2_s2');
save('train_data_p.mat','train_data_p');
save('train_label.mat','train_label');
save('train_list.mat','train_list');

save('test_data_1_s2.mat','test_data_1_s2');
save('test_data_2_s2.mat','test_data_2_s2');
save('test_data_p.mat','test_data_p');
save('test_label.mat','test_label');
save('test_list.mat','test_list');

save('val_data_1_s2.mat','val_data_1_s2');
save('val_data_2_s2.mat','val_data_2_s2');
save('val_data_p_all.mat','val_data_p');
save('val_label.mat','val_label');
save('val_list.mat','val_list');
% 
% check = val_label;
% check(val_label==0)=4;
% abc = (check==4);
% idx = find(abc == 1);
% randsamp = randi(length(idx),1,300);
% index = idx(randsamp);

cls_num=7;
val_label_256=zeros(256,1);
val_data_1_s2_256=zeros(256,4096);
val_data_2_s2_256=zeros(256,4096);
val_data_p_256=zeros(256,4096);
count=1;
for i=1:cls_num+1
    if i-1==2 || i-1==3
        tem_1_s2 = val_data_1_s2(val_label==(i-1),:);
        tem_2_s2 = val_data_2_s2(val_label==(i-1),:);
        val_data_1_s2_256(count:count+size(tem_1_s2,1)-1,:)=tem_1_s2;
        val_data_2_s2_256(count:count+size(tem_2_s2,1)-1,:)=tem_2_s2;
        
        tem_p = val_data_p(val_label==(i-1),:);
        val_data_p_256(count:count+size(tem_p,1)-1,:)=tem_p;
        val_label_256(count:count+size(tem_1_s2,1))=i-1;
        count=count+size(tem_1_s2,1);
    else
        tem_1_s2 = val_data_1_s2(val_label==(i-1),:);
        tem_2_s2 = val_data_2_s2(val_label==(i-1),:);
        radsmp = randi(size(tem_1_s2,1),1,35);
        val_data_1_s2_256(count:count+34,:)=tem_1_s2(radsmp,:);
        val_data_2_s2_256(count:count+34,:)=tem_2_s2(radsmp,:);
        
        tem_p = val_data_p(val_label==(i-1),:);
        val_data_p_256(count:count+34,:)=tem_p(radsmp,:);
        val_label_256(count:count+34)=i-1;
        count=count+35;
    end
    
    clear tem_s2;
    clear tem_p;
end