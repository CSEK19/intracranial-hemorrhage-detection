function [accuracyLogit, precisionLogit, recallLogit, f1scoreLogit] = LogitBoostModel(XTrain, YTrain, XTest, YTest)
    % GuassianModel - A static function to train and test a Logit Boost 
    % Classifier.
    %
    % Syntax:
    %   [accuracyLogit, precisionLogit, recallLogit, f1scoreLogit] = LogitBoostModel(XTrain, YTrain, XTest, YTest);
    %
    % Input:
    %   XTrain - Features of training set.
    %   YTrain - Labels of training set.
    %   XTest - Features of testing set.
    %   YTest - Labels of testing set.
    %
    % Output:
    %   accuracyLogit, precisionLogit, recallLogit, f1scoreLogit

    % Train LogitBoost model
    logitBoostModel = fitcensemble(XTrain, YTrain, 'Method', 'LogitBoost');
    
    % Predict using LogitBoost model
    YTestPredLogit = predict(logitBoostModel, XTest);
    
    % Evaluate the performance
    confMatLogit = confusionmat(YTest, YTestPredLogit);
    disp('Confusion Matrix for LogitBoost:');
    disp(confMatLogit);
    
    accuracyLogit = sum(diag(confMatLogit)) / sum(confMatLogit(:));
    fprintf('Accuracy of LogitBoost: %.2f%%\n', accuracyLogit * 100);
    
    % Calculate precision, recall, and F1-score for LogitBoost model
    TP_Logit = diag(confMatLogit);
    FP_Logit = sum(confMatLogit, 1)' - TP_Logit;
    FN_Logit = sum(confMatLogit, 2) - TP_Logit;
    precisionLogit = TP_Logit ./ (TP_Logit + FP_Logit);
    recallLogit = TP_Logit ./ (TP_Logit + FN_Logit);
    f1scoreLogit = 2 * (precisionLogit .* recallLogit) ./ (precisionLogit + recallLogit); 
end