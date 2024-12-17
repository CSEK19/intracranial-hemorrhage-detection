function [accuracyGaussianNB, precisionGaussianNB, recallGaussianNB, f1scoreGaussianNB] = GuassianModel(XTrain, YTrain, XTest, YTest)
    % GuassianModel - A static function to train and test a Guassian Model 
    % Classifier.
    %
    % Syntax:
    %   [accuracyGaussianNB, precisionGaussianNB, recallGaussianNB, f1scoreGaussianNB] = GuassianModel(XTrain, YTrain, XTest, YTest);
    %
    % Input:
    %   XTrain - Features of training set.
    %   YTrain - Labels of training set.
    %   XTest - Features of testing set.
    %   YTest - Labels of testing set.
    %
    % Output:
    %   accuracyGaussianNB, precisionGaussianNB, recallGaussianNB, f1scoreGaussianNB

    % Train Gaussian Naive Bayes model
    gaussianNBModel = fitcnb(XTrain, YTrain);
    
    % Predict using Gaussian Naive Bayes model
    YTestPredGaussianNB = predict(gaussianNBModel, XTest);
    
    % Evaluate the performance for Gaussian Naive Bayes
    confMatGaussianNB = confusionmat(YTest, YTestPredGaussianNB);
    disp('Confusion Matrix for Gaussian Naive Bayes:');
    disp(confMatGaussianNB);
    
    accuracyGaussianNB = sum(diag(confMatGaussianNB)) / sum(confMatGaussianNB(:));
    fprintf('Accuracy of Gaussian Naive Bayes: %.2f%%\n', accuracyGaussianNB * 100);
    
    % Calculate precision, recall, and F1-score for Gaussian Naive Bayes
    TP_GaussianNB = diag(confMatGaussianNB);
    FP_GaussianNB = sum(confMatGaussianNB, 1)' - TP_GaussianNB;
    FN_GaussianNB = sum(confMatGaussianNB, 2) - TP_GaussianNB;
    precisionGaussianNB = TP_GaussianNB ./ (TP_GaussianNB + FP_GaussianNB);
    recallGaussianNB = TP_GaussianNB ./ (TP_GaussianNB + FN_GaussianNB);
    f1scoreGaussianNB = 2 * (precisionGaussianNB .* recallGaussianNB) ./ (precisionGaussianNB + recallGaussianNB);
end