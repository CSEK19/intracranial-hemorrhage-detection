outputFolder = fullfile(pwd, 'Outputs');
if ~isfolder(outputFolder)
    mkdir(outputFolder);
else
    files = dir(fullfile(outputFolder, '*'));
    for k = 1:length(files)
        fileName = files(k).name;
        if ~strcmp(fileName, '.') && ~strcmp(fileName, '..')
            delete(fullfile(outputFolder, fileName));
        end
    end
end

dataFolder = fullfile(pwd, 'Dataset');
fileList = dir(fullfile(dataFolder, '*.mat'));

for k = 1:length(fileList)
    [~, dataName, ~] = fileparts(fileList(k).name);
    fprintf('Training and Testing models on %s dataset.\n', dataName);
    dataFile = fullfile(dataFolder, fileList(k).name);
    outputFile = fullfile(pwd, 'Outputs', ['Output_', dataName, '.txt']);
    tableFile = fullfile(pwd, 'Outputs', ['ResultsTable_', dataName, '.xlsx']);

    diary(outputFile);

    [inputs, targets] = LoadDataset(dataFile);
    
    rng(121); % For reproducibility
    
    % Partition the data for training and testing (80% training, 20% testing)
    cv = cvpartition(targets, 'HoldOut', 0.2);
    trainIdx = training(cv);
    testIdx = test(cv);
    
    XTrain = inputs(trainIdx, :);
    YTrain = targets(trainIdx, :);
    XTest = inputs(testIdx, :);
    YTest = targets(testIdx, :);
    
    [accuracyRUS, precisionRUS, recallRUS, f1scoreRUS] = RUSBoostModel(XTrain, YTrain, XTest, YTest);
    
    [accuracyAda, precisionAda, recallAda, f1scoreAda] = AdaBoostModel(XTrain, YTrain, XTest, YTest);
    
    [accuracyLogit, precisionLogit, recallLogit, f1scoreLogit] = LogitBoostModel(XTrain, YTrain, XTest, YTest);
    
    [accuracyNN, precisionNN, recallNN, f1scoreNN] = NNModel(XTrain, YTrain, XTest, YTest, 122);
    
    [accuracySVM, precisionSVM, recallSVM, f1scoreSVM] = SVMModel(XTrain, YTrain, XTest, YTest, 123);
    
    [accuracyGaussianNB, precisionGaussianNB, recallGaussianNB, f1scoreGaussianNB] = GuassianModel(XTrain, YTrain, XTest, YTest);
    
    ModelNames = {'RUSBoost'; 'AdaBoost'; 'LogitBoost'; 'Neural Network'; 'SVM'; 'Gaussian Naive Bayes'};
    AccuracyArray = [accuracyRUS; accuracyAda; accuracyLogit; accuracyNN; accuracySVM; accuracyGaussianNB];
    PrecisionArray = [mean(precisionRUS); mean(precisionAda); mean(precisionLogit); mean(precisionNN); mean(precisionSVM); mean(precisionGaussianNB)];
    RecallArray = [mean(recallRUS); mean(recallAda); mean(recallLogit); mean(recallNN); mean(recallSVM); mean(recallGaussianNB)];
    F1ScoreArray = [mean(f1scoreRUS); mean(f1scoreAda); mean(f1scoreLogit); mean(f1scoreNN); mean(f1scoreSVM); mean(f1scoreGaussianNB)];
    
    % Create a table with all metrics
    resultsTable = table(ModelNames, AccuracyArray, PrecisionArray, RecallArray, F1ScoreArray, ...
                         'VariableNames', {'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'});
    
    % Write table to txt file
    writetable(resultsTable, tableFile);
    
    % Disable diary
    diary off;
end

