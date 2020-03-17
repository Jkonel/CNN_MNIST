function image = vecter2image(dat,num)
%% 将dat中第num列转换成方阵image
[row,~] = size(dat);

row1 = sqrt(row);
col1 = sqrt(row);

image = zeros(row1,col1);

for i = 1:row1
    for j = 1:col1
        image(j,i) = dat(j+28*(i-1),num);
    end
end

