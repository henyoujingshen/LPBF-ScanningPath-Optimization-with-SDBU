function path_tempmatrix = to_tempmatrix(n_try,xn_elements,yn_elements,frame_gap,every_n_frame)
    %TO_TEMPMATRIX n_try为第n次仿真
    n_elements=xn_elements*yn_elements;
    temp_txt=sprintf('I:\\SLM\\TEMP%d.txt',n_try);
    new_folder=sprintf('I:\\SLM\\tempmatrix_%d',n_try);
    mkdir(new_folder);
    temp_element=readmatrix(temp_txt);
    %先整合一遍，8行变一行，使node的数据变为element的数据
%     frame=0;
%     temp_element=[];
%     for i=1:length(temp)
%         if mod(i,n_nodes)==0
%             frame=frame+1;
%             adds_up=0;
%             for line=1:n_nodes
%                 adds_up=adds_up+temp(i+line,3);
%             end
%             temp_element(frame,1)=temp(i,1);
%             temp_element(frame,2)=temp(i,2);
%             temp_element(frame,3)=adds_up/n_nodes;
%         end
%     end
    % 再继续对temp_element进行操作
    path_tempmatrix=[];
    temperature=zeros(50,50);
    %跳过铺粉时间
    start_step=0+0.02/frame_gap;
    t=0;
    %首先先把每一个step的数据存在一个cell中
    %这里是扫描过程中使用的参数
    for i=start_step*n_elements:(n_elements+start_step)*n_elements
    %for i=1:length(temp_element)
        if mod(i-start_step*n_elements,every_n_frame*n_elements)==1
        %if mod(i,every_n_frame*n_elements)==1
            t=t+1;
            for col=1:xn_elements
                for row=1:yn_elements
                    temperature(row,col)=temp_element(i+row-1+(col-1)*xn_elements,3);
                end
            end
            %上下颠倒矩阵
            temperature=flipud(temperature);
            excel=sprintf('%s\\temperautre_at_%dsteps.csv',new_folder,t);
            writematrix(temperature,excel)
            path_tempmatrix=strvcat(path_tempmatrix,excel);
        end
    end
end

