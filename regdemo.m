x = dataset(:,1);
y = dataset(:,2);

figure;
plot(x, y, '.', 'MarkerSize', 20);
xlabel('Input');
ylabel('Output');
grid on;
hold on;

% nosakâm attiecîgâs pakâpes polinoma parametrus a
degree = 2;
a = linreg(x, y, degree);

% veicam prognozi visos xq punktos, lai uzzîmçtu modeïa grafiku
xq = (min(x):0.01:max(x))'; %zq = linspace(min(z), max(z), 100)';
yHat = linreg_predict(xq, a, degree);
plot(xq, yHat, '-');

% veicam prognozi vienâ xq punktâ un uzzîmçjam to tai paðâ attçlâ
xq = 3.5;
yHat = linreg_predict(xq, a, degree);
fprintf('Pie x = %.1f prognozçtais y ir %.1f\n', xq, yHat);
plot(xq, yHat, '*r');

% modeïa novçrtçðana apmâcîbas kopâ
yHat = linreg_predict(x, a, degree);
[SAE, MAE, SSE, MSE, RMSE, R2] = evaluate_model(y, yHat);
fprintf('SAE=%.2f, MAE=%.2f, SSE=%.2f, MSE=%.2f, RMSE=%.2f, R2=%.2f\n', SAE, MAE, SSE, MSE, RMSE, R2);
