%% ����˵��
%          1���ػ���pooling������ƽ��2*2
%          2����������˵����
%                           ����㣺28*28��y��
%                           ��һ�㣺24*24�������*20
%                           tanh
%                           �ڶ��㣺12*12��pooling��*20
%                           �����㣺100(ȫ����)
%                           ���Ĳ㣺10(softmax)
%          3������ѵ�����ֲ���800�����������鲿�ֲ���100������
clear;
clc;close all;
%% ���ݶ���
train_num = 5000;%ѵ����������
test_num = 200;
%%%%%���ݶ��������%%%%%
images_train = loadMNISTImages('train-images.idx3-ubyte');
dat_train = images_train(:,1:train_num);
labels_train = loadMNISTLabels('train-labels.idx1-ubyte');
labels_train = labels_train(1:train_num,:);

dat_test = loadMNISTImages('t10k-images.idx3-ubyte');
dat_test = dat_test(:,1:test_num);
labels_test = loadMNISTLabels('t10k-labels.idx1-ubyte');
labels_test = labels_test(1:test_num,:);

%% �����ʼ��
layer_c1_num=20;
layer_s1_num=20;
layer_f1_num=100;
layer_output_num=10;
%Ȩֵ��������
yita=0.01;
%bias��ʼ��
bias_c1=(2*rand(1,20)-ones(1,20))/sqrt(20);
bias_f1=(2*rand(1,100)-ones(1,100))/sqrt(20);
%����˳�ʼ��
[kernel_c1,kernel_f1]=init_kernel(layer_c1_num,layer_f1_num);
%pooling�˳�ʼ��
pooling_a=ones(2,2)/4;
%ȫ���Ӳ��Ȩֵ
weight_f1=(2*rand(20,100)-ones(20,100))/sqrt(20);
weight_output=(2*rand(100,10)-ones(100,10))/sqrt(100);
disp('�����ʼ�����......');

%% ��ʼ����ѵ��
disp('��ʼ����ѵ��......');
for iter=1:5
    fprintf('iter = %d\n',iter);
    for n = 1:train_num/10    %1000����
        fprintf('n = %d\n',n);
        for m=0:9
%             fprintf('m = %d\n',m);
            %��ȡ����
            %train_data=imread(strcat(num2str(m),'_',num2str(n),'.bmp'));
            train_data = vecter2image(dat_train,10*(n-1)+m+1);
            train_data=double(train_data);
            
            % ȥ��ֵ
            %train_data=wipe_off_average(train_data);
            %ǰ�򴫵�,��������1
            for k=1:layer_c1_num
                state_c1(:,:,k)=convolution(train_data,kernel_c1(:,:,k));
                %���뼤������
                state_c1(:,:,k)=tanh(state_c1(:,:,k)+bias_c1(1,k));
                %����pooling1
                state_s1(:,:,k)=pooling(state_c1(:,:,k),pooling_a);
            end
            %(ȫ���Ӳ�)
            %����f1��
            [state_f1_pre,state_f1_temp]=convolution_f1(state_s1,kernel_f1,weight_f1);
            %���뼤������
            for nn=1:layer_f1_num
                state_f1(1,nn)=tanh(state_f1_pre(:,:,nn)+bias_f1(1,nn));
            end
            %����softmax�㣨BP��
            for nn=1:layer_output_num
                output(1,nn)=exp(state_f1*weight_output(:,nn))/sum(exp(state_f1*weight_output));
            end
            %% �����㲿��
            Error_cost=-output(1,labels_train(10*(n-1)+m+1)+1);
            %         if (Error_cost<-0.98)
            %             break;
            %         end
            %% ������������
            [kernel_c1,kernel_f1,weight_f1,weight_output,bias_c1,bias_f1]=CNN_upweight(yita,Error_cost,labels_train(10*(n-1)+m+1),train_data,...
                state_c1,state_s1,...
                state_f1,state_f1_temp,...
                output,...
                kernel_c1,kernel_f1,weight_f1,weight_output,bias_c1,bias_f1);
        end
    end
end
%%

%save('CNNѵ������');
disp('����ѵ����ɣ���ʼ����......');

count=0;
for n=1:test_num/10
    for m=0:9
        %��ȡ����
%        train_data=imread(strcat(num2str(m),'_',num2str(n),'.bmp'));
        train_data = vecter2image(dat_test,10*(n-1)+m+1);
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
        if (classify==labels_test(10*(n-1)+m+1)+1)
            count=count+1;
        end
        fprintf('��ʵ����Ϊ%d  ������Ϊ%d  ����ֵΪ%d \n',labels_test(10*(n-1)+m+1),classify-1,p);
    end
end

fprintf('׼ȷ�ʣ�%d \n',(count/test_num));


