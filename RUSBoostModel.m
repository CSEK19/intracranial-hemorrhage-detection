function [accuracyRUS, precisionRUS, recallRUS, f1scoreRUS] = RUSBoostModel(XTrain, YTrain, XTest, YTest)
    % GuassianModel - A static function to train and test a Neural Network 
    % Classifier.
    %
    % Syntax:
    %   [accuracyRUS, precisionRUS, recallRUS, f1scoreRUS] = RUSBoostModel(XTrain, YTrain, XTest, YTest);
    %
    % Input:
    %   XTrain - Features of training set.
    %   YTrain - Labels of training set.
    %   XTest - Features of testing set.
    %   YTest - Labels of testing set.
    %
    % Output:
    %   accuracyRUS, precisionRUS, recallRUS, f1scoreRUS

    N = size(XTrain, 1); % Number of observations in the training sample

    tRUS = templateTree('MaxNumSplits', N); % Example weak learner template for RUSBoost
    rusBoostModel = fitcensemble(XTrain, YTrain, 'Method', 'RUSBoost', ...
                                 'NumLearningCycles', 1000, ...
                                 'Learners', tRUS, ...
                                 'LearnRate', 0.1);
    
    % Predict test data and calculate accuracy for RUSBoost
    predictionsRUS = predict(rusBoostModel, XTest);
    accuracyRUS = sum(predictionsRUS == YTest) / numel(YTest);
    fprintf('Accuracy of RUSBoost: %.2f%%\n', accuracyRUS * 100);
    
    % Generate and display confusion matrix for RUSBoost
    confMatRUS = confusionmat(YTest, predictionsRUS);
    disp('Confusion Matrix for RUSBoost:');
    disp(confMatRUS);
    
    % Extract true positives, false positives, and false negatives for RUSBoost model
    TP_RUS = diag(confMatRUS); 
    FP_RUS = sum(confMatRUS, 1)' - TP_RUS; 
    FN_RUS = sum(confMatRUS, 2) - TP_RUS; 
    % Calculate precision, recall, and F1-score for RUSBoost model
    precisionRUS = TP_RUS ./ (TP_RUS + FP_RUS); 
    recallRUS = TP_RUS ./ (TP_RUS + FN_RUS);    
    f1scoreRUS = 2 * (precisionRUS .* recallRUS) ./ (precisionRUS + recallRUS); 
    
    % Handle cases where precision or recall is NaN due to division by zero
    precisionRUS(isnan(precisionRUS)) = 0;
    recallRUS(isnan(recallRUS)) = 0;
    f1scoreRUS(isnan(f1scoreRUS)) = 0;
end