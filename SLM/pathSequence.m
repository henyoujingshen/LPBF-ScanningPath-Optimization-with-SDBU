function [PP,RP] = pathSequence(index,num,iffig)
%% 初始参数
X = 1; Y = 1; RP = []; PP = []; t = 0;
%% 棋盘格扫描时序
n = 5; writematrix(n,['n',num2str(n),'.txt']);
writematrix(index,['index',num2str(n),'.txt']);
plot_X = 1:1:n; plot_Y = 1:1:n;
figure; imagesc(index);
for i = 1:size(plot_Y,2)
    for j = 1:size(plot_X,2)
        text(j,i, num2str(index(i,j)),'HorizontalAlignment','center');
    end
end
colormap summer;axis equal;axis off;
colorbar('AxisLocation','out','Location','east','Ticks',[1,n*n],...
    'TickLabels',{'start','end'});
% saveas(gcf,['chessboard',num2str(num),'.tiff']);
% saveas(gcf,['chessboard',num2str(num),'.svg']);
%% %%%%%%%%%%%%%%%%%%%%%%%%%% 逐层扫描 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ii = 1:1
    n_layer = ii; Z = n_layer*0.1/1000;
    %% %%%%%%%%%%%%%%%%%%%%%%%%%% 铺粉 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    r = [t,-2/1000,Y*n/2/1000,Z,1]; RP = [RP;r]; t = t+0.02;                         % 铺粉时间假设为0.02s，实际1s左右，对散热影响不大
    r = [t,(X*n+2)/1000,Y*n/2/1000,Z,0]; RP = [RP;r]; t = RP(end,1);
    %% %%%%%%%%%%%%%%%%%%%%%%% 单个棋盘格扫描路径 %%%%%%%%%%%%%%%%%%%%%%
    for i = 1:n*n
        [row,col] = find(index == i);
        X_offset = (col-1)*X; Y_offset = (n-row)*Y;
        if mod((row+col),2) == 0
            angle = 0;
        else
            angle = 90;
        end
        [PP] = pathSingle(X_offset,Y_offset,X,Y,angle,Z,t,PP);
        t = PP(end,1);
    end
end

%%  %%%%%%%%%%%%%%%%%%%%%%%%%% 路径预览 %%%%%%%%%%%%%%%%%%%%%%%%%%%
if iffig == 1
    % laser 轨迹
    figure
    axis equal; xlim([0 X*n]); ylim([0 Y*n]);
    rectangle('Position',[0 0 X*n Y*n],'edgecolor','r','linewidth',1.5)
    hold on
    for j = 1:n-1
        plot([X*j X*j],[0 Y*n],'r','linewidth',1); hold on
        plot([0 X*n],[Y*j Y*j],'r','linewidth',1); hold on;
    end
    pause(0.3)
    for i = 1:length(PP(:,2))-1
        if PP(i,5) == 0
            plot([PP(i,2),PP(i+1,2)]*1000,[PP(i,3),PP(i+1,3)]*1000,...
                'w','linewidth',1)
        else
            plot([PP(i,2),PP(i+1,2)]*1000,[PP(i,3),PP(i+1,3)]*1000,...
                'k','linewidth',1)
        end
        pause(0.01); hold on
    end

    % roller 轨迹
    figure
    axis equal; xlim([0 X*n]); ylim([0 Y*n]);
    rectangle('Position',[0 0 X*n Y*n],'edgecolor','r','linewidth',1.5)
    hold on
    for j = 1:n-1
        plot([X*j X*j],[0 Y*n],'r','linewidth',1); hold on
        plot([0 X*n],[Y*j Y*j],'r','linewidth',1); hold on;
    end
    pause(0.3)
    for i = 1:length(RP(:,2))-1
        if RP(i,5) == 0
            plot([RP(i,2),RP(i+1,2)]*1000,[RP(i,3),RP(i+1,3)]*1000,...
                'w','linewidth',1.5)
        else
            plot([RP(i,2),RP(i+1,2)]*1000,[RP(i,3),RP(i+1,3)]*1000,...
                'k','linewidth',1.5)
        end
        pause(0.3);hold on
    end
end
end