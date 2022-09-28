f_name = 'C:\Users\camalas.DEACNET\Project\Planet_S1S2_training_data';

dest_rgb = 'C:\Users\camalas.DEACNET\Project\Planet_S1S2_training_data\RGB\';
dest_lbl = 'C:\Users\camalas.DEACNET\Project\Planet_S1S2_training_data\Threshold_Label\';
dest = 'C:\Users\camalas.DEACNET\Project\Planet_S1S2_training_data\Thr_Labels_final\';
wtr_thr = 0;
act_thr = 0;
inc_thr = 0.15;
no_veg_thr = 0.199;  % -1 to 0.199
low_veg_thr = 0.75;   % 0.2 to 0.5
% high_veg = 0.501 to 1.0

cmap = zeros(7, 3);
cmap = [0,0,0;...           % black for others
    1, 1, 0; ...            % Yellow for 1 active pond
    1, 0, 0; ...            % Red for 2  transition pond
    0, 1, 0;...             % Green for 3 inactive pond
    0, 1, 1;...             % for 4 forest
    1, 0, 1;...             % for 5 agriculture
    1, 1, 1;...             % White for 6 no vegitation
    0, 0, 1];               % blue for 7 build-up                       

names = 's2_test_20210807_3173x3224.tif';
S1 = imread(strcat(f_name,'\','s1_test_20210808_3173x3224.tif'));
S2 = imread(strcat(f_name,'\',names));
P = imread(strcat(f_name,'\','planet_20210807\4772862_1933110_2021-08-07_2455_BGRN_Analytic_toar_clip.tif'));

S2 = uint8(double(S2)/4000.0*255);
% S2 = rescale(S2,0,255);
% P = rescale(P,0,255); %Blue,Green,Red,NIR
P = uint8(double(P)/4500.0*255);
 
    new_rgb_s2 = S2(:,:,[4,3,2]);
    new_rgb_p = P(:,:,[3,2,1]);
%     I_rgb = uint8(double(new_rgb)/4000.0*255);
    I_rgb_s2 = uint8(rescale(new_rgb_s2,0,255)); 
    I_rgb_p = uint8(new_rgb_p);
    imwrite(I_rgb_s2,strcat(dest_rgb,'S2_rgb.tif'));
    imwrite(I_rgb_p,strcat(dest_rgb,'P_rgb.tif'));

    WI_s2 = (double(S2(:,:,3))-double(S2(:,:,11)))./(double(S2(:,:,3))+double(S2(:,:,11)));    % SWIR-Green ratio for water index
    YI_s2 = (double(I_rgb_s2(:,:,2))-double(I_rgb_s2(:,:,1)))./(double(I_rgb_s2(:,:,2))+double(I_rgb_s2(:,:,1))); % Green-Red ratio for pond color
    VI_s2 = (double(S2(:,:,8))-double(S2(:,:,4)))./(double(S2(:,:,8))+double(S2(:,:,4)));           %NIR-red ratio for vegitation index
    BI_s2 = (double(S2(:,:,11))-double(S2(:,:,8)))./(double(S2(:,:,11))+double(S2(:,:,8))); % NDBI = (SWIR - NIR) / (SWIR + NIR) for urban/build-up index
    
    WI_p = (double(P(:,:,2))-double(P(:,:,4)))./(double(P(:,:,2))+double(P(:,:,4)));    % NIR-Green ratio 
    VI_p = (double(P(:,:,4))-double(P(:,:,3)))./(double(P(:,:,4))+double(P(:,:,3)));    % NIR-Red for vegi
    
    BI_s1 = (S1(:,:,1)+S1(:,:,2)); % S1
    
    
    water_mask = WI_s2>wtr_thr;
    active_mask = YI_s2<=act_thr;
    inactive_mask = YI_s2>inc_thr;
    transition_mask = YI_s2<inc_thr & YI_s2>act_thr;
    built_up_mask = BI_s2>0 & uint8(rescale(BI_s1,0,255))>195;
    
    active_pond = water_mask & active_mask;
    transition_pond = water_mask & transition_mask;
    inactive_pond = water_mask & inactive_mask;
    
    no_veg = (VI_s2 < no_veg_thr)&(VI_p < no_veg_thr)&~(water_mask);
    low_veg = (VI_s2 <= low_veg_thr) & (VI_p <= 0.6) & ~(water_mask);
    high_veg = (VI_s2 >= low_veg_thr) &(VI_p >= 0.6);
    
    new_lbl = zeros(size(I_rgb_s2,1),size(I_rgb_s2,2));
    new_lbl(active_pond) = 1;
    new_lbl(transition_pond) = 2;
    new_lbl(inactive_pond) = 3;       
    new_lbl(high_veg) = 4;            % Forest area
    new_lbl(low_veg) = 5;           % Agricaltural area
    new_lbl(no_veg) = 6;          % not defined area
    new_lbl(built_up_mask) = 7;      % Built-up area
    
%     imwrite(uint8(new_lbl),strcat(dest_lbl,'S2_rgb_4000_thr.png'));
    new_lbl = imread('C:\Users\camalas.DEACNET\Project\Planet_S1S2_training_data\Thr_Labels_final\s2_test_20210807_3173x3224_correct.png');
     I = new_lbl; %double(new_lbl);
    BW = bwlabel(I,4);
    new = zeros(size(I,1),size(I,2));
    
    reg  = regionprops(BW,I,{'Area','PixelValues','PixelList'});
    
    for j = 2:length(reg)
        if reg(j).Area > 2
            mask = zeros(size(I,1),size(I,2));
            
            values = unique(reg(j).PixelValues);
            list = [reg(j).PixelList(:,2) reg(j).PixelList(:,1)];   
            for l = 1:length(list)
                mask(list(l,1),list(l,2)) = 1;
            end
%             figure, imshow(mask);
            for v = 1:length(values)
                sum_val(v) = length(find(reg(j).PixelValues==values(v)));
            end
            max_val = values(find(sum_val==max(sum_val)));
            if length(max_val) >1 %&& length(find(max_val==2))>0
                if max_val(1)==3 && max_val(2)==2
                    new(mask==1) = 2;
                else if max_val(1)==4 && max_val(2)==5
                        new(mask==1) = 5;
                    end
                end
            else
                new(mask==1) = max_val(1);
            end
            
        end
        clear sum_val
        clear max_val
        close all
    end
    %Check if the label is good or not

    fig = figure,
    ax1 = subplot(1,2,1)
    imshow(I_rgb_p), title('RGB Image');
    ax2 = subplot(1,2,2)
    imshow(new_lbl+1,cmap), title('Threshold Labeling');
%     ax3 = subplot(1,3,3)
    colorbar(gca);
    linkaxes([ax1,ax2],'xy');
    set(gcf, 'Position', get(0, 'Screensize'));
    
     prompt = 'Do you accept this? y/n';
    str = input(prompt,'s');
    if str == 'n'
%         figure,imshow(I_rgb_p);
        flag='y';
        
        while(flag=='y')
            figure,imshow(I_rgb_p);
            roi = drawfreehand('Color','r','Multiclick',1);
            prompt = 'Which label is this? (1-Active/2-Transition/3-Inactive/4-Forest/5-Agriculture/6-No Veg/7-Build-up)';
            lbl = str2double(input(prompt,'s'));
            mask2=zeros(size(I,1),size(I,2));
            mask3=zeros(size(I,1),size(I,2));
%             pos = round(roi.Position);
%             for l = 1:length(pos)
%                 if pos(l,2)>0 && pos(l,2)<size(I,1) && pos(l,1)>0 && pos(l,1)<size(I,2)
%                     mask2(pos(l,2),pos(l,1)) = 1;
%                 end
%             end
%             mask2=imfill(mask2);
            mask2=roi.createMask();
            figure,imshow(label2rgb(mask2)) 
            % where mask and rio is intersect, label them on new label
%             mask3=mask2&(new_lbl==0);
%             mask3=imfill(mask3,'holes');
%             figure,imshow(label2rgb(mask3)) 
            new_lbl(mask2)=lbl;
%             figure,imshow(label2rgb(new_lbl)) 
            prompt = 'Do you want to continue? y/n';
            flag = input(prompt,'s');
        end
    end
    
    imwrite(uint8(new_lbl),strcat(dest,strrep(names,'.tif','_correct.png')));
