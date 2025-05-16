%% 流程处理
clc,clear,close all;
clearvars

%% 计算数据step合适的时间间隔(最小公倍数)
frame_gap=0.0002;
path_gap=0.0002;
yn_elements=50;
xn_elements=50;
n_nodes=8;
n_elements=xn_elements*yn_elements;
%step_gap=lcm(frame_gap,path_gap);
step_gap=0.0002;
every_n_frame=step_gap/frame_gap;
every_n_path=step_gap/path_gap;

%% 先生成deposition matrix
n_try=3;
path_depomatrix=to_depomatrix(n_try);

%% 再生成对应的temp并储存
path_tempmatrix = to_tempmatrix(n_try,xn_elements,yn_elements,frame_gap,every_n_frame);


%% 最后生成有pair两个对应地址的root.txt,里面每一行都是数据和label的csv
root=[path_depomatrix path_tempmatrix];

writematrix(root,[sprintf('root_data%d.txt',n_try)],'delimiter',',');

