function [accuracyAda, precisionAda, recallAda, f1scoreAda] = AdaBoostModel(XTrain, YTrain, XTest, YTest)
    % GuassianModel - A static function to train and test a Ada Boost 
    % Classifier.
    %
    % Syntax:
    %   [accuracyAda, precisionAda, recallAda, f1scoreAda] = AdaBoostModel(XTrain, YTrain, XTest, YTest);
    %
    % Input:
    %   XTrain - Features of training set.
    %   YTrain - Labels of training set.
    %   XTest - Features of testing set.
    %   YTest - Labels of testing set.
    %
    % Output:
    %   accuracyAda, precisionAda, recallAda, f1scoreAda

    % Train AdaBoost model
    adaBoostModel = fitcensemble(XTrain, YTrain, 'Method', 'AdaBoostM1');
    
    % Predict using AdaBoost model
    YTestPredAda = predict(adaBoostModel, XTest);
    
    % Evaluate the performance
    confMatAda = confusionmat(YTest, YTestPredAda);
    disp('Confusion Matrix for AdaBoost:');
    disp(confMatAda);
    
    accuracyAda = sum(diag(confMatAda)) / sum(confMatAda(:));
    fprintf('Accuracy of AdaBoost: %.2f%%\n', accuracyAda * 100);
    
    % Calculate precision, recall, and F1-score for AdaBoost model
    TP_Ada = diag(confMatAda);
    FP_Ada = sum(confMatAda, 1)' - TP_Ada;
    FN_Ada = sum(confMatAda, 2) - TP_Ada;
    precisionAda = TP_Ada ./ (TP_Ada + FP_Ada);
    recallAda = TP_Ada ./ (TP_Ada + FN_Ada);
    f1scoreAda = 2 * (precisionAda .* recallAda) ./ (precisionAda + recallAda);
end