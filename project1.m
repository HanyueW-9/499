%%
% *Hanyue Wang(NAU ID: wh259)， Xiaobai Li(NAU ID: xl253)*      
%
% GitHub link : https://github.com/HanyueW-9/499/blob/master/project1.m
%
% *MATLAB Lab*
%
% *Project 1*
%
% *CS499*


X = importdata('SAheart.data');
% import SAheart data set
data = X.data;
data1 = zeros(size(data,1),6);
textdata = X.textdata;

[m,n] = size(textdata);
% use size function to calculate the size of data(famhist has how many
% columns)

i = 2;
% set the column variable to pass the headlind
while(i<=m)
% use while function count the column range
    
    if(textdata(i,6) == "Present")
        textdata(i,6)= cellstr("1");
        % transform the string to cell by using cellstr()
        i = i + 1;
    else
        textdata(i,6) = cellstr( "0");
        i = i + 1;
    end
    % transform the "Present" and "Absent" into binary indicator
end

% transform textdata type from "str" to "int"
for i =1:size(data,1)
    for j = 1:6
        data1(i,j) = str2num(textdata{i+1,j});
    end
end
% concat data1 and data
data  = [data1,data]

output_SAheart = data(:,end);
% out put the least column in the data set

% data scale
data = scale(data(:,1:end-1));
% concat x and Y
data = [data,output_SAheart];

% split data 70% for training 30% for testing.
[trainX,testX,validationX] = divideblock(data', .6, .2, .2); % 60% for training 20% for testing. 20% for validation
trainData = trainX.'; 
testData = testX.';
validationData = validationX.';

result  = zeros(3,2);
% output 0/1 in trainData testData validationData
result(1,1)=sum( trainData(:,end)==0);  result(1,2)=sum( trainData(:,end)==1); 
result(2,1)=sum( testData(:,end)==0);  result(2,2)=sum( testData(:,end)==1); 
result(3,1)= sum( validationData(:,end)==0);result(3,2)=sum( validationData(:,end)==1); 

% Gradientdescent and plot
[weight,error,vali_error] = Gradientdescent(trainData(:,1:end-1),trainData(:,end),validationData(:,1:end-1),validationData(:,end),1000,0.00008)
figure;
index = find(error==min(error))
hold on;
plot(error);
plot(vali_error)
legend('train','validation');
plot(index,error(index),'ro')
hold on;
index = find(vali_error==min(vali_error))
hold on;
plot(index,vali_error(index),'ro')
title('SAheart data set');
% error in trainData testData validationData
vali_label = [ones(size(validationData(:,1:end-1),1),1),validationData(:,1:end-1)]*weight;
train_label =[ones(size(trainData(:,1:end-1),1),1),trainData(:,1:end-1)]*weight;
test_label = [ones(size(testData(:,1:end-1),1),1),testData(:,1:end-1)]*weight;

result1 = Labelresult(vali_label,validationData(:,end),train_label,trainData(:,end),test_label,testData(:,end));

hold off;
plot(error/(vali_error+error))
xlabel('maxIterations');
ylabel('error rate');
% Y = importdata('spam.data');
% % import SAheart data set
% 
% % separate the data into two cells
% output_spam = Y(:,end);
% 
% set_size = 0:0.1:4;
% 
% [trainY,testY,validationY] = divideblock(Y.', .6, .2, .2); % 70% for training 30% for testing.
% trainData1 = trainY.'; 
% testData1 = testY.';
% validationData1 = validationY.';
Y = importdata('spam.data');
data2 = Y;
output_spam = data2(:,end);
% data scale
data2 = scale(data2(:,1:end-1));
% concat x and Y
data2 = [data2,output_spam];

[trainY,testY,validationY] = divideblock(data2', .6, .2, .2); % 60% for training 20% for testing. 20% for validation
trainData2 = trainY.'; 
testData2 = testY.';
validationData2 = validationY.';

result  = zeros(3,2);
% output 0/1 in trainData testData validationData
result(1,1)=sum( trainData2(:,end)==0);  result(1,2)=sum( trainData2(:,end)==1); 
result(2,1)=sum( testData2(:,end)==0);  result(2,2)=sum( testData2(:,end)==1); 
result(3,1)= sum( validationData2(:,end)==0);result(3,2)=sum( validationData2(:,end)==1); 

[weight,error,vali_error] = Gradientdescent(trainData2(:,1:end-1),trainData2(:,end),validationData2(:,1:end-1),validationData2(:,end),1000,0.00008)
figure;
index = find(error==min(error))
hold on;
plot(error);
plot(vali_error)
legend('train','validation');
plot(index,error(index),'ro')
hold on;
index = find(vali_error==min(vali_error))
hold on;
plot(index,vali_error(index),'ro')
title('Spam data set');

% error in trainData testData validationData
vali_label2 = [ones(size(validationData2(:,1:end-1),1),1),validationData2(:,1:end-1)]*weight;
train_label2 =[ones(size(trainData2(:,1:end-1),1),1),trainData2(:,1:end-1)]*weight;
test_label2 = [ones(size(testData2(:,1:end-1),1),1),testData2(:,1:end-1)]*weight;

result2 = Labelresult(vali_label2,validationData2(:,end),train_label2,trainData2(:,end),test_label2,testData2(:,end))

hold off;
plot(vali_error/error)
xlabel('maxIterations');
ylabel('error rate');


Z = importdata('zip.train');
data3 = Z;
output_zip = data3(:,end);
% data scale
data3 = scale(data3(:,1:end-1));
% concat x and Y
data3 = [data3,output_zip];

[trainZ,testZ,validationZ] = divideblock(data3', .6, .2, .2);  % 60% for training 20% for testing. 20% for validation
trainData3 = trainZ.'; 
testData3 = testZ.';
validationData3 = validationZ.';

result  = zeros(3,2);
% output 0/1 in trainData testData validationData
result(1,1)=sum( trainData3(:,end)==0);  result(1,2)=sum( trainData3(:,end)==1); 
result(2,1)=sum( testData3(:,end)==0);  result(2,2)=sum( testData3(:,end)==1); 
result(3,1)= sum( validationData3(:,end)==0);result(3,2)=sum( validationData3(:,end)==1); 

[weight,error,vali_error] = Gradientdescent(trainData3(:,1:end-1),trainData3(:,end),validationData3(:,1:end-1),validationData3(:,end),1000,0.00008)
figure;
index = find(error==min(error))
hold on;
plot(error);
plot(vali_error)
legend('train','validation');
plot(index,error(index),'ro')
hold on;
index = find(vali_error==min(vali_error))
hold on;
plot(index,vali_error(index),'ro')
title('Zip data set');

% error in trainData testData validationData
vali_label3 = [ones(size(validationData3(:,1:end-1),1),1),validationData3(:,1:end-1)]*weight;
train_label3 =[ones(size(trainData3(:,1:end-1),1),1),trainData3(:,1:end-1)]*weight;
test_label3 = [ones(size(testData3(:,1:end-1),1),1),testData3(:,1:end-1)]*weight;

result3 = Labelresult(vali_label3,validationData3(:,end),train_label3,trainData3(:,end),test_label3,testData3(:,end))

hold off;
plot(error/(vali_error+error))
xlabel('maxIterations');
ylabel('error rate');
% function weightMatrix = Gradientdescent(X,y,maxIterations,stepSize)
% weightVector = zeros(size(X,2)+1,1);
% weightMatrix = zeros(size(X,2)+1,maxIterations);
% X = [ones(size(X,1),1),X];
%     for k = 1 : maxIterations
%         gradient = X'*(X*weightVector-y);
%         weightVector = weightVector - stepSize*gradient;
%         weightMatrix(:,k) = weightVector;
%     end
% end
function  result = Labelresult(vali_label,vali_y,train_label,train_y,test_label,test_y)
    result = zeros(3,1);
    for i = 1:size(vali_label,1)
        if(vali_label(i)>=0.5)
            vali_label(i) = 1;
        else
            vali_label(i) = 0;
        end
    end
    
    for i = 1:size(train_label,1)
        if(train_label(i)>=0.5)
            train_label(i) = 1;
        else
            train_label(i) = 0;
        end
    end
    
     for i = 1:size(test_label,1)
        if(test_label(i)>=0.5)
            test_label(i) = 1;
        else
            test_label(i) = 0;
        end
     end
    result(1,1) = norm(vali_y - vali_label)/size(vali_label,1);
    result(2,1) = norm(train_y - train_label)/size(train_label,1);
     result(3,1) = norm(test_y - test_label)/size(test_label,1);
     
    
end

function [W,error,vali_error] = Gradientdescent(X,Y,vali_X,vali_Y,maxIterations,stepSize)
    error = zeros(maxIterations,1);
    vali_error = zeros(maxIterations,1);
    tX = X;
    W = 0.1*ones(size(X,2)+1,1);
    X = [ones(size(X,1),1),X];
    vali_X = [ones(size(vali_X,1),1),vali_X];
    k = 0;
    while true
        detaW = zeros(size(tX,2)+1,1);
        O = X*W;
        detaW = detaW + stepSize*X'*(Y - O);
        W = W + detaW;
        k = k+1;
        if 1/2*( norm(Y - X*W) )^2 < 0.05 || k> maxIterations 
            break;
        end
        fprintf('iterator times %d, error %f\n',k,1/2*( norm(Y - X*W) )^2);
        error(k,1) =1/2*( norm(Y - X*W) )^2;
        vali_error(k,1) =1/2*( norm(vali_Y - vali_X*W) )^2;
    end
end


function A = scale(input)
    M=mean(input);
    S=std(input);
    A = (input-repmat(M,size(input,1),1));
    for i =1:size(A,2)
        A(:,i) = A(:,i)/S(i)
    end
end

function weightMatrix = GD(X,y,maxIterations,stepSize)
weightVector = zeros(size(X,2),1);
weightMatrix = zeros(size(X,2),maxIterations);
gradient = [];

for k = 1:maxIterations
    for m = 1:size(X,1)
        h = 1./(1+exp(-X(m,:) * weightVector));
        tmp = (h - y(m)) * X(m,:);
        gradient = [gradient;tmp];
    end  
    gradient = mean(gradient);
    weightVector = weightVector - stepSize*gradient';
    weightMatrix(:,k) = weightVector;
end
