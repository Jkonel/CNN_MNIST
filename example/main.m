%% 程序说明
%          1、池化（pooling）采用平均2*2
%          2、网络结点数说明：
%                           输入层：28*28（y）
%                           第一层：24*24（卷积）*20
%                           tanh
%                           第二层：12*12（pooling）*20
%                           第三层：100(全连接)
%                           第四层：10(softmax)
%          3、网络训练部分采用800个样本，检验部分采用100个样本
clear;
clc;close all;
%% 数据读入
train_num = 5000;%训练集样本数
test_num = 200;
%%%%%数据读入初处理%%%%%
images_train = loadMNISTImages('train-images.idx3-ubyte');
dat_train = images_train(:,1:train_num);
labels_train = loadMNISTLabels('train-labels.idx1-ubyte');
labels_train = labels_train(1:train_num,:);

dat_test = loadMNISTImages('t10k-images.idx3-ubyte');
dat_test = dat_test(:,1:test_num);
labels_test = loadMNISTLabels('t10k-labels.idx1-ubyte');
labels_test = labels_test(1:test_num,:);

%% 网络初始化
layer_c1_num=20;
layer_s1_num=20;
layer_f1_num=100;
layer_output_num=10;
%权值调整步进
yita=0.01;
%bias初始化
bias_c1=(2*rand(1,20)-ones(1,20))/sqrt(20);
bias_f1=(2*rand(1,100)-ones(1,100))/sqrt(20);
%卷积核初始化
[kernel_c1,kernel_f1]=init_kernel(layer_c1_num,layer_f1_num);
%pooling核初始化
pooling_a=ones(2,2)/4;
%全连接层的权值
weight_f1=(2*rand(20,100)-ones(20,100))/sqrt(20);
weight_output=(2*rand(100,10)-ones(100,10))/sqrt(100);
disp('网络初始化完成......');

%% 开始网络训练
disp('开始网络训练......');
for iter=1:5
    fprintf('iter = %d\n',iter);
    for n = 1:train_num/10    %1000样本
        fprintf('n = %d\n',n);
        for m=0:9
%             fprintf('m = %d\n',m);
            %读取样本
            %train_data=imread(strcat(num2str(m),'_',num2str(n),'.bmp'));
            train_data = vecter2image(dat_train,10*(n-1)+m+1);
            train_data=double(train_data);
            
            % 去均值
            %train_data=wipe_off_average(train_data);
            %前向传递,进入卷积层1
            for k=1:layer_c1_num
                state_c1(:,:,k)=convolution(train_data,kernel_c1(:,:,k));
                %进入激励函数
                state_c1(:,:,k)=tanh(state_c1(:,:,k)+bias_c1(1,k));
                %进入pooling1
                state_s1(:,:,k)=pooling(state_c1(:,:,k),pooling_a);
            end
            %(全连接层)
            %进入f1层
            [state_f1_pre,state_f1_temp]=convolution_f1(state_s1,kernel_f1,weight_f1);
            %进入激励函数
            for nn=1:layer_f1_num
                state_f1(1,nn)=tanh(state_f1_pre(:,:,nn)+bias_f1(1,nn));
            end
            %进入softmax层（BP）
            for nn=1:layer_output_num
                output(1,nn)=exp(state_f1*weight_output(:,nn))/sum(exp(state_f1*weight_output));
            end
            %% 误差计算部分
            Error_cost=-output(1,labels_train(10*(n-1)+m+1)+1);
            %         if (Error_cost<-0.98)
            %             break;
            %         end
            %% 参数调整部分
            [kernel_c1,kernel_f1,weight_f1,weight_output,bias_c1,bias_f1]=CNN_upweight(yita,Error_cost,labels_train(10*(n-1)+m+1),train_data,...
                state_c1,state_s1,...
                state_f1,state_f1_temp,...
                output,...
                kernel_c1,kernel_f1,weight_f1,weight_output,bias_c1,bias_f1);
        end
    end
end
%%

%save('CNN训练参数');
disp('网络训练完成，开始检验......');

count=0;
for n=1:test_num/10
    for m=0:9
        %读取样本
%        train_data=imread(strcat(num2str(m),'_',num2str(n),'.bmp'));
        train_data = vecter2image(dat_test,10*(n-1)+m+1);
        train_data=double(train_data);
        % 去均值
        % train_data=wipe_off_average(train_data);
        %前向传递,进入卷积层1
        for k=1:layer_c1_num
            state_c1(:,:,k)=convolution(train_data,kernel_c1(:,:,k));
            %进入激励函数
            state_c1(:,:,k)=tanh(state_c1(:,:,k)+bias_c1(1,k));
            %进入pooling1
            state_s1(:,:,k)=pooling(state_c1(:,:,k),pooling_a);
        end
        %进入f1层
        [state_f1_pre,state_f1_temp]=convolution_f1(state_s1,kernel_f1,weight_f1);
        %进入激励函数
        for nn=1:layer_f1_num
            state_f1(1,nn)=tanh(state_f1_pre(:,:,nn)+bias_f1(1,nn));
        end
        %进入softmax层
        for nn=1:layer_output_num
            output(1,nn)=exp(state_f1*weight_output(:,nn))/sum(exp(state_f1*weight_output));
        end
        [p,classify]=max(output);
        if (classify==labels_test(10*(n-1)+m+1)+1)
            count=count+1;
        end
        fprintf('真实数字为%d  网络标记为%d  概率值为%d \n',labels_test(10*(n-1)+m+1),classify-1,p);
    end
end

fprintf('准确率：%d \n',(count/test_num));


