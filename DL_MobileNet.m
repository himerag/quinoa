% ================================================== ===
% Convolutional neural network analysis - 07/27/2022
% CNN = MobileNet
% Dr Wilson Castro
% Analysis of quinoa grains and foreign materials
% ====================================================
% clear memory and display
clc, clear
% setting constants
Tamano=224;
Tipo = {'RGB' 'LAB' 'HSV' 'YCC' 'GRAY'};
Ntrainings=20;
NumEpoc=6;
ItxEp=148;
CM_RGB=zeros(17,17,Ntrainings);
AC_LO_RGB=zeros(Ntrainings, ItxEp*NumEpoc,2);
CM_LAB=zeros(17,17,Ntrainings);
AC_LO_LAB=zeros(Ntrainings, ItxEp*NumEpoc,2);
CM_HSV=zeros(17,17,Ntrainings);
AC_LO_HSV=zeros(Ntrainings, ItxEp*NumEpoc,2);
CM_YCC=zeros(17,17,Ntrainings);
AC_LO_YCC=zeros(Ntrainings, ItxEp*NumEpoc,2);
CM_GRAY=zeros(17,17,Ntrainings);
AC_LO_GRAY=zeros(Ntrainings, ItxEp*NumEpoc,2);
TimeElap = zeros(1,Ntrainings); % tiempo por cada iteracion
for NTipo=1:length(Tipo)
    display(strcat('En evaluacion = ',Tipo{NTipo}))
    % concadenating folder names
    carpeta =strcat(cd,'\ImageDataSet224x224\Imagenes_',num2str(Tamano),'_',Tipo{NTipo});
    % loading images datasets
    imds = imageDatastore(carpeta, ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');
    % loading the network
    net=mobilenetv2;
    lgraph = layerGraph(net); 
    % setting the red for leraning transfer
    numClasses= numel(categories(imds.Labels));
    nFC = fullyConnectedLayer(numClasses, ...
        'Name','FC17',...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    lgraph = replaceLayer(lgraph,'Logits',nFC);
    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'ClassificationLayer_Logits',newClassLayer);
    % Training and statistical metrics calculation 
    f = waitbar(0, 'Calculando');
    
    for i=1:Ntrainings
    
        % data split
        [imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
        
       % options for network setting
        options = trainingOptions('sgdm', ...
        'MiniBatchSize',80, ...
        'MaxEpochs',NumEpoc, ...
        'InitialLearnRate',0.0001, ...
        'Shuffle','every-epoch', ...
        'ValidationData',imdsValidation, ...
        'ValidationFrequency',50, ...
        'ExecutionEnvironment', 'auto',...
        'Verbose',false, ...
        'Plots','none');
    
        tic
        % network training
        [netTransfer, info] = trainNetwork(imdsTrain,lgraph,options);
        TimeElap(1,i)=toc;
        
        % validation step
        [YPred,scores] = classify(netTransfer,imdsValidation);
        
        % confusion matrix determining
        if NTipo==1
            CM_RGB(:,:,i) = confusionmat(imdsValidation.Labels,YPred);
            AC_LO_RGB(i,:,1)=info.TrainingAccuracy; % Accuracy
            AC_LO_RGB(i,:,2)=info.TrainingLoss; % Loss
            T_RGB=TimeElap;
        elseif NTipo==2
            CM_LAB(:,:,i) = confusionmat(imdsValidation.Labels,YPred);
            AC_LO_LAB(i,:,1)=info.TrainingAccuracy; % Accuracy
            AC_LO_LAB(i,:,2)=info.TrainingLoss; % Loss
            T_LAB=TimeElap;
        elseif NTipo==3
            CM_HSV(:,:,i) = confusionmat(imdsValidation.Labels,YPred);
            AC_LO_HSV(i,:,1)=info.TrainingAccuracy; % Accuracy
            AC_LO_HSV(i,:,2)=info.TrainingLoss; % Loss      
            T_HSV=TimeElap;
        elseif NTipo==4
            CM_YCC(:,:,i) = confusionmat(imdsValidation.Labels,YPred);
            AC_LO_YCC(i,:,1)=info.TrainingAccuracy; % Accuracy
            AC_LO_YCC(i,:,2)=info.TrainingLoss; % Loss    
            T_YCC=TimeElap;
        elseif NTipo==5
            CM_GRAY(:,:,i) = confusionmat(imdsValidation.Labels,YPred);
            AC_LO_GRAY(i,:,1)=info.TrainingAccuracy; % Accuracy
            AC_LO_GRAY(i,:,2)=info.TrainingLoss; % Loss  
            T_GRAY=TimeElap;
        end
        
        waitbar(i/Ntrainings, f, sprintf('Progress: %d %%', floor(i/Ntrainings*100)));
    
        % saving results
        if NTipo==1
            save CM_RGB CM_RGB;
            save AC_LO_RGB AC_LO_RGB
            save T_RGB T_RGB
        elseif NTipo==2
            save CM_LAB CM_LAB;
            save AC_LO_LAB AC_LO_LAB
            save T_LAB T_LAB
        elseif NTipo==3
            save CM_HSV CM_HSV;
            save AC_LO_HSV AC_LO_HSV
            save T_HSV T_HSV
        elseif NTipo==4
            save CM_YCC CM_YCC;
            save AC_LO_YCC AC_LO_YCC
            save T_YCC T_YCC
        elseif NTipo==5
            save CM_GRAY CM_GRAY;
            save AC_LO_GRAY AC_LO_GRAY
            save T_GRAY T_GRAY
        end       
        
        
    end
    close(f)
      
    display(strcat('Culminado=',Tipo{NTipo}))
end
