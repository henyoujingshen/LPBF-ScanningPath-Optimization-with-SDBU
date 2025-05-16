%% %%%%%%%%%%%%%%%%%%%%%%% 单个棋盘格扫描路径 %%%%%%%%%%%%%%%%%%%%%%
% X_offset0 = 0; Y_offset0 = 0; X0 = 1; Y0 = 0; angle = 90; n_layer = 1; t0 = 0;
function [PP] = pathSingle(X_offset0,Y_offset0,X0,Y0,angle,Z,t0,PP)
% 参数初始化
v = 500/1000; h = 0.1/1000; LaserPower = 100;                              % v, h, LaserPower 分别为扫描速度、搭接间距、激光功率,SI单位
X = X0/1000; Y = Y0/1000;                                                    % 单个棋盘格尺寸，SI单位
layer_thickness = 0.05/1000;
X_offset = X_offset0/1000; Y_offset = Y_offset0/1000;
t = round(t0,5);                                                            % 初始化
X_begin = X_offset+h/2; X_end = X-h/2+X_offset;
Y_begin = Y_offset+h/2; Y_end = Y-h/2+Y_offset;

% 棋盘格扫描
if angle == 0
    Y_y = Y_begin; n_tracks = Y/h;
    for k = 1:n_tracks
        if mod(k,2)==1
            a = [t,X_begin,Y_y,Z,LaserPower]; t = t+X/v; t = round(t,5);       % 激光扫描部分
            b = [t,X_end,Y_y,Z,0]; t = t+h/v; t = round(t,5);
        else
            a = [t,X_end,Y_y,Z,LaserPower]; t = t+X/v; t = round(t,5);
            b = [t,X_begin,Y_y,Z,0]; t = t+h/v; t = round(t,5);
        end

        %b = [t,X_end,Y_y,Z,0]; t = t+h/v; t = round(t,5);                  % 激光移动不扫描
        Y_y = Y_y+h; PP = [PP;a;b];
    end
end
if angle == 90
    X_x = X_begin; n_tracks = X/h;                                       
    for k = 1:n_tracks
        if mod(k,2)==1
            a = [t,X_x,Y_begin,Z,LaserPower]; t =t+Y/v; t = round(t,5);
            b = [t,X_x,Y_end,Z,0]; t = t+h/v; t = round(t,5);
        else
            a = [t,X_x,Y_end,Z,LaserPower]; t =t+Y/v; t = round(t,5);
            b = [t,X_x,Y_begin,Z,0]; t = t+h/v; t = round(t,5);
        end
        
        X_x = X_x+h; PP = [PP;a;b];
    end
end
end

