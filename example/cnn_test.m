function [p,classify] = cnn_test(train_data,kernel_c1,kernel_f1,weight_f1,weight_output,bias_c1,bias_f1)
% 网络初始化
layer_c1_num=20;
layer_s1_num=20;
layer_f1_num=100;
layer_output_num=10;
pooling_a=ones(2,2)/4;

%load('CNN训练参数.mat');
%读取样本
%        train_data=imread(strcat(num2str(m),'_',num2str(n),'.bmp'));
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
classify=classify-1;

end

