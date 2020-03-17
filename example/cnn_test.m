function [p,classify] = cnn_test(train_data,kernel_c1,kernel_f1,weight_f1,weight_output,bias_c1,bias_f1)
% �����ʼ��
layer_c1_num=20;
layer_s1_num=20;
layer_f1_num=100;
layer_output_num=10;
pooling_a=ones(2,2)/4;

%load('CNNѵ������.mat');
%��ȡ����
%        train_data=imread(strcat(num2str(m),'_',num2str(n),'.bmp'));
train_data=double(train_data);
% ȥ��ֵ
% train_data=wipe_off_average(train_data);
%ǰ�򴫵�,��������1
for k=1:layer_c1_num
    state_c1(:,:,k)=convolution(train_data,kernel_c1(:,:,k));
    %���뼤������
    state_c1(:,:,k)=tanh(state_c1(:,:,k)+bias_c1(1,k));
    %����pooling1
    state_s1(:,:,k)=pooling(state_c1(:,:,k),pooling_a);
end
%����f1��
[state_f1_pre,state_f1_temp]=convolution_f1(state_s1,kernel_f1,weight_f1);
%���뼤������
for nn=1:layer_f1_num
    state_f1(1,nn)=tanh(state_f1_pre(:,:,nn)+bias_f1(1,nn));
end
%����softmax��
for nn=1:layer_output_num
    output(1,nn)=exp(state_f1*weight_output(:,nn))/sum(exp(state_f1*weight_output));
end
[p,classify]=max(output);
classify=classify-1;

end

