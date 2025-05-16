clc;clear all
%%%%% 棋盘格扫描、往复、扫完一层完全冷却100s %%%%%%
%% 生成各种激光扫描路径,分为n*n的棋盘格,m为总共进行几次
n=5;
for m=1:3
    num = 1;                                                                   % 编号
    iffig = 1;                                                                 % 是否画路径图，1-画，0-不画
    % index = randperm(n*n); index = reshape(index,[n,n]);
    % 随机扫描序列
    A=(1:n^2);
    randomindex_A=randperm(n^2);
    B=A(randomindex_A);
    index=[];
    for p=1:n
       vector=[];
       for q=1:n
           vector=[vector B((p-1)*n+q)];
       end
       index=[index;vector];
    end

    [PP,RP] = pathSequence(index,num,iffig);
    t_max = PP(end,1); writematrix(t_max,['t_max5.txt']);
    writematrix(RP,['RP5.txt'],'delimiter',',');
    writematrix(PP,['PP5.txt'],'delimiter',',');
    %close all
%% 更新AM model的激光扫描路径
   % system('abaqus cae noGUI=SLM.py')             % 模型几何参数不变，此步不用进行

%% 提交计算文件
system('abaqus job=Job_TEMP interactive')
%system('abaqus job=Job_TEMP_cooling interactive')


%% 读取计算结果NT
    arg_cmd=sprintf('abaqus cae noGUI=Output.py -- %d',m)
    system(arg_cmd)



%% 以下只是将表示路径存为scv，文件的读取和矩阵转化,反解出沉积矩阵的工作在data_prepare中进行
    excel=sprintf('I:\\SLM\\TEMP%d.csv',m)
    writematrix(index,excel)
end