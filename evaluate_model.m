function [SAE, MAE, SSE, MSE, RMSE, R2, AIC, AICC, BIC] = evaluate_model(y, yHat, k)
% funkcija aprçíina kïûdu kritçriju vçrtîbas
    n = size(y,1);
    SAE = sum(abs(y - yHat));
    MAE = SAE / n;
    SSE = sum((y - yHat) .^ 2);
    MSE = SSE / n;
    RMSE = sqrt(MSE);
    SSEtot = sum((y - mean(y)) .^ 2);
    R2 = 1 - SSE / SSEtot;
    AIC = n * log(MSE) + 2 * k;
    AICC = AIC + 2 * k * (k + 1) / (n - k - 1);
    BIC = n * log(MSE) + k * log(n);
end
