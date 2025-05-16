function path_depomatrix=to_depomatrix(n_try)
    %% 存depo矩阵为时间和depo状态
    new_folder=sprintf('D:\\SLM\\depomatrix_%d',n_try);
    path_eccel=sprintf('D:\\SLM\\TEMP%d.csv',n_try);
    mkdir(new_folder);
    path_depomatrix=[];
    depo=zeros(50,50);
    t = 0;       % 初始化
    n_tracks=10; %目前的假设是轨迹正好通过沉积元素中心，且棋盘为正方形，n_track是横向以及纵向的元素数量
    index=readmatrix(path_eccel);
    n=5;
    for i = 1:n*n
        [row,col] = find(index == i);
        x_offset = (col-1)*10; y_offset = (row-1)*10;
        if mod((row+col),2) == 0
            Y_y = y_offset;
            for k = 1:n_tracks
                Y_y = Y_y+1;
                if mod(k,2)==1
                    x_track=x_offset;
                    for kk = 1:n_tracks
                        t =t+1;
                        x_track=x_track+1;
                        depo(Y_y,x_track)=0.000001;
                        for m=1:50
                            for n=1:50
                                if depo(m,n)~=0
                                    depo(m,n)=depo(m,n)+1;
                                end
                            end
                        end
                        excel=sprintf('%s\\deposition_at_%dsteps.csv',new_folder,t);
                        writematrix(depo,excel)
                        path_depomatrix=strvcat(path_depomatrix,excel);
                    end
                else
                    x_track=x_offset+n_tracks;
                    for kk = 1:n_tracks
                        t =t+1;
                        depo(Y_y,x_track)=0.000001;
                        for m=1:50
                            for n=1:50
                                if depo(m,n)~=0
                                    depo(m,n)=depo(m,n)+1;
                                end
                            end
                        end
                        excel=sprintf('%s\\deposition_at_%dsteps.csv',new_folder,t);
                        writematrix(depo,excel)
                        path_depomatrix=strvcat(path_depomatrix,excel);
                        x_track=x_track-1;
                    end
                end
    
            end
        else
            X_x = x_offset;
            for k = 1:n_tracks
                X_x = X_x+1;
                if mod(k,2)==1
                    y_track=y_offset;
                    for kk = 1:n_tracks
                        y_track=y_track+1;
                        t =t+1;
                        %% 激活
                        depo(y_track,X_x)=0.0000001;
                        %% 所有沉积矩阵的其他位置+1
                        for m=1:50
                            for n=1:50
                                if depo(m,n)~=0
                                    depo(m,n)=depo(m,n)+1;
                                end
                            end
                        end
                        excel=sprintf('%s\\deposition_at_%dsteps.csv',new_folder,t);
                        writematrix(depo,excel)
                        path_depomatrix=strvcat(path_depomatrix,excel);
                    end
                else
                    y_track=y_offset+n_tracks;
                    for kk = 1:n_tracks
                        t =t+1;
                        depo(y_track,X_x)=0.0000001;
                        %% 所有沉积矩阵的其他位置+1
                        for m=1:50
                            for n=1:50
                                if depo(m,n)~=0
                                    depo(m,n)=depo(m,n)+1;
                                end
                            end
                        end
                        excel=sprintf('%s\\deposition_at_%dsteps.csv',new_folder,t);
                        writematrix(depo,excel)
                        path_depomatrix=strvcat(path_depomatrix,excel);
                        y_track=y_track-1;
                    end
                end
            end
        end
    end
end