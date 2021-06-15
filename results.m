t0 = datetime('2020-01-09 00:00:00', 'Format', 'uuuu-MM-dd HH:mm:ss');
t1 = datetime('2020-03-30 00:00:00', 'Format', 'uuuu-MM-dd HH:mm:ss');
t2 = datetime('2020-04-06 00:00:00', 'Format', 'uuuu-MM-dd HH:mm:ss');
t3 = datetime('2020-07-13 00:00:00', 'Format', 'uuuu-MM-dd HH:mm:ss');
t4 = datetime('2020-07-20 00:00:00', 'Format', 'uuuu-MM-dd HH:mm:ss');
t5 = datetime('2020-09-14 00:00:00', 'Format', 'uuuu-MM-dd HH:mm:ss');
t6 = datetime('2020-09-21 00:00:00', 'Format', 'uuuu-MM-dd HH:mm:ss');
t7 = datetime('2020-11-23 00:00:00', 'Format', 'uuuu-MM-dd HH:mm:ss');
t8 = datetime('2020-11-30 00:00:00', 'Format', 'uuuu-MM-dd HH:mm:ss');

load('Date', 'Date')


result0 = find(finalresults(:, 1).Variables == t0) - 1;
result1 = find(finalresults(:, 1).Variables == t1);
result2 = find(finalresults(:, 1).Variables == t2) - 1;
result3 = find(finalresults(:, 1).Variables == t3);
result4 = find(finalresults(:, 1).Variables == t4) - 1;
result5 = find(finalresults(:, 1).Variables == t5);
result6 = find(finalresults(:, 1).Variables == t6) - 1;
result7 = find(finalresults(:, 1).Variables == t7);
result8 = find(finalresults(:, 1).Variables == t8) - 1;

demand_phase0 = finalresults(1:result0, 2);
demandpred_phase0_covid = finalresults(1:result0, 18);
demandpred_phase0_nocovid = finalresultsnoCOVID(1:result0, 18);
date_phase0 = finalresults(1:result0, 1);
error_phase0 = finalresults(1:result0, 19);

demand_phase1 = finalresults(result1:result2, 2);
demandpred_phase1_covid = finalresults(result1:result2, 18);
demandpred_phase1_nocovid = finalresultsnoCOVID(result1:result2, 18);
date_phase1 = finalresults(result1:result2, 1);
error_phase1_covid = finalresults(result1:result2, 19);
error_phase1_nocovid = finalresultsnoCOVID(result1:result2, 19);

demand_phase2 = finalresults(result3:result4, 2);
demandpred_phase2_covid = finalresults(result3:result4, 18);
demandpred_phase2_nocovid = finalresultsnoCOVID(result3:result4, 18);
date_phase2 = finalresults(result3:result4, 1);
error_phase2_covid = finalresults(result3:result4, 19);
error_phase2_nocovid = finalresultsnoCOVID(result3:result4, 19);

demand_phase3 = finalresults(result5:result6, 2);
demandpred_phase3_covid = finalresults(result5:result6, 18);
demandpred_phase3_nocovid = finalresultsnoCOVID(result5:result6, 18);
date_phase3 = finalresults(result5:result6, 1);
error_phase3_covid = finalresults(result5:result6, 19);
error_phase3_nocovid = finalresultsnoCOVID(result5:result6, 19);

demand_phase4 = finalresults(result7:result8, 2);
demandpred_phase4_covid = finalresults(result7:result8, 18);
demandpred_phase4_nocovid = finalresultsnoCOVID(result7:result8, 18);
date_phase4 = finalresults(result7:result8, 1);
error_phase4_covid = finalresults(result7:result8, 19);
error_phase4_nocovid = finalresultsnoCOVID(result7:result8, 19);


y1 = movmax(finalresults(:, 39).Variables, 24);
y2 = movmax(finalresultsnoCOVID(:, 19).Variables, 24);
error_covid = [];
error_nocovid = [];

for i = 1:24:8747
    error_covid = [error_covid y1(i)];
end

for i = 1:24:8747
    error_nocovid = [error_nocovid y2(i)];
end
error_covid = [error_covid error_covid(365)];
error_nocovid = [error_nocovid error_nocovid(365)];

err_0_comb = error_covid(2:9);
err_1_comb = [error_covid(90:97); error_nocovid(90:97)];
err_2_comb = [error_covid(195:202); error_nocovid(195:202)];
err_3_comb = [error_covid(258:265); error_nocovid(258:265)];
err_4_comb = [error_covid(328:335); error_nocovid(328:335)];



figure(1);
hold on
p = plot(date_phase0.Variables, demand_phase0.Variables, 'LineWidth', 1.5, 'DisplayName', 'Actual Load Demand');
plot(date_phase0.Variables, demandpred_phase0_covid.Variables, 'LineWidth', 1.5, 'DisplayName', 'Predicted Load Demand');
title('Data and Model Prediction Phase 0', 'FontSize', 16);
xlabel('Date', 'FontSize', 14);
ylabel('Load Demand (MW)', 'FontSize', 14);
ylim([20000 50000]);
ax = ancestor(p, 'axes');
ax.YAxis.Exponent = 0;
ax.FontSize = 12;
ytickformat('%.0f');
grid on;
legend ('show', 'FontSize', 14);
width = 900; %sets width and height of the graph
height = 500;
set(gcf,'position',[10,10,width,height])
saveas(gcf,'pred_phase0','epsc')
hold off;

figure(2);
hold on;
p = plot(date_phase1.Variables, demand_phase1.Variables, 'LineWidth', 1.5, 'DisplayName', 'Actual Load Demand');
plot(date_phase1.Variables, demandpred_phase1_covid.Variables, 'LineWidth', 1.5, 'DisplayName', 'Predicted Load Demand (Adjusted Model)');
plot(date_phase1.Variables, demandpred_phase1_nocovid.Variables, 'LineWidth', 1.5, 'DisplayName', 'Predicted Load Demand (Unadjusted Model)');
title('Data and Model Prediction Phase 1', 'FontSize', 16);
xlabel('Date', 'FontSize', 14);
ylabel('Load Demand (MW)', 'FontSize', 14);
ylim([15000 40000]);
ax = ancestor(p, 'axes');
ax.YAxis.Exponent = 0;
ax.FontSize = 12;
ytickformat('%.0f');
grid on;
legend ('show', 'FontSize', 14);
width = 900; %sets width and height of the graph
height = 500;
set(gcf,'position',[10,10,width,height])
saveas(gcf,'pred_phase1','epsc')
hold off;

figure(3);
hold on;
p = plot(date_phase2.Variables, demand_phase2.Variables, 'LineWidth', 1.5, 'DisplayName', 'Actual Load Demand');
plot(date_phase2.Variables, demandpred_phase2_covid.Variables, 'LineWidth', 1.5, 'DisplayName', 'Predicted Load Demand (Adjusted Model)');
plot(date_phase2.Variables, demandpred_phase2_nocovid.Variables, 'LineWidth', 1.5, 'DisplayName', 'Predicted Load Demand (Unadjusted Model)');
title('Data and Model Prediction Phase 2', 'FontSize', 16);
xlabel('Date', 'FontSize', 14);
ylabel('Load Demand (MW)', 'FontSize', 14);
ylim([15000 40000]);
ax = ancestor(p, 'axes');
ax.YAxis.Exponent = 0;
ax.FontSize = 12;
ytickformat('%.0f');
grid on;
legend ('show', 'FontSize', 14);
width = 900; %sets width and height of the graph
height = 500;
set(gcf,'position',[10,10,width,height])
saveas(gcf,'pred_phase2','epsc')
hold off;

figure(4);
hold on;
p = plot(date_phase3.Variables, demand_phase3.Variables, 'LineWidth', 1.5, 'DisplayName', 'Actual Load Demand');
plot(date_phase3.Variables, demandpred_phase3_covid.Variables, 'LineWidth', 1.5, 'DisplayName', 'Predicted Load Demand (Adjusted Model)');
plot(date_phase3.Variables, demandpred_phase3_nocovid.Variables, 'LineWidth', 1.5, 'DisplayName', 'Predicted Load Demand (Unadjusted Model)');
title('Data and Model Prediction Phase 3', 'FontSize', 16);
ylabel('Load Demand (MW)', 'FontSize', 14);
ylim([15000 42000])
ax = ancestor(p, 'axes');
ax.YAxis.Exponent = 0;
ax.FontSize = 12;
ytickformat('%.0f');
grid on;
legend ('show', 'FontSize', 14);
width = 900; %sets width and height of the graph
height = 500;
set(gcf,'position',[10,10,width,height])
saveas(gcf,'pred_phase3','epsc')
hold off;

figure(5);
hold on;
p = plot(date_phase4.Variables, demand_phase4.Variables, 'LineWidth', 1.5, 'DisplayName', 'Actual Load Demand');
plot(date_phase4.Variables, demandpred_phase4_covid.Variables, 'LineWidth', 1.5, 'DisplayName', 'Predicted Load Demand (Adjusted Model)');
plot(date_phase4.Variables, demandpred_phase4_nocovid.Variables, 'LineWidth', 1.5, 'DisplayName', 'Predicted Load Demand (Unadjusted Model)');
title('Data and Model Prediction Phase 4', 'FontSize', 16);
xlabel('Date', 'FontSize', 14);
ylabel('Load Demand (MW)', 'FontSize', 14);
ylim([20000 50000]);
ax = ancestor(p, 'axes');
ax.YAxis.Exponent = 0;
ax.FontSize = 12;
ytickformat('%.0f');
grid on;
legend ('show', 'FontSize', 14);
width = 900; %sets width and height of the graph
height = 500;
set(gcf,'position',[10,10,width,height])
saveas(gcf,'pred_phase4','epsc')
hold off;

figure(6);
boxplot(finalresults(:, 19).Variables, finalresults(:, 12).Variables)
title('Breakdown of Forecast Error by Day', 'FontSize', 16)
xlabel('Day of Week', 'FontSize', 14)
ylabel('Percentage Error',  'FontSize', 14)
xticklabels({'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'})
ylim([-0.5 40])
ax = ancestor(p, 'axes');
ax.YAxis.Exponent = 0;
ax.FontSize = 12;
grid on
width = 700; %sets width and height of the graph
height = 300;
set(gcf,'position',[10,10,width,height])
saveas(gcf,'boxplot_covid','epsc')


figure(7);
boxplot(finalresultsnoCOVID(:, 19).Variables, finalresultsnoCOVID(:, 12).Variables)
title('Breakdown of Forecast Error by Day', 'FontSize', 16)
xlabel('Day of Week', 'FontSize', 14)
ylabel('Percentage Error',  'FontSize', 14)
xticklabels({'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'})
ylim([-0.5 40])
ax = ancestor(p, 'axes');
ax.YAxis.Exponent = 0;
ax.FontSize = 12;
grid on
width = 700; %sets width and height of the graph
height = 300;
set(gcf,'position',[10,10,width,height])
saveas(gcf,'boxplot_nocovid','epsc')
%}



figure(8);
hold on
p = bar(Date(2:9), err_0_comb);
set(p, {'DisplayName'}, {'Forecasting Errors'}')
ax = gca;
ax.FontSize = 12;
title('Max Daily Forecasting Errors in Phase 0', 'FontSize', 16);
xlabel('Date', 'FontSize', 14);
ylabel('Percentage Error', 'FontSize', 14);
ytickformat('%.0f');
grid on;
legend ('show', 'FontSize', 14);
width = 900; %sets width and height of the graph
height = 500;
set(gcf,'position',[10,10,width,height])
saveas(gcf,'error_phase0','epsc')
hold off;


figure(9);
hold on;
p = bar(Date(90:97), err_1_comb);
set(p, {'DisplayName'}, {'Adjusted Model Forecasting Errors','Unadjusted Model Forecasting Errors'}')
ax = gca;
ax.FontSize = 12;
title('Max Daily Forecasting Errors in Phase 1', 'FontSize', 16);
xlabel('Date', 'FontSize', 14);
ylabel('Percentage Error', 'FontSize', 14);
ytickformat('%.0f');
grid on;
legend ('show', 'FontSize', 14);
width = 900; %sets width and height of the graph
height = 500;
set(gcf,'position',[10,10,width,height])
saveas(gcf,'error_phase1','epsc')
hold off;

figure(10);
hold on;
p = bar(Date(195:202), err_2_comb);
set(p, {'DisplayName'}, {'Adjusted Model Forecasting Errors','Unadjusted Model Forecasting Errors'}')
ax = gca;
ax.FontSize = 12;
title('Max Daily Forecasting Errors in Phase 2', 'FontSize', 16);
xlabel('Date', 'FontSize', 14);
ylabel('Percentage Error', 'FontSize', 14);
ytickformat('%.0f');
grid on;
legend ('show', 'FontSize', 14);
width = 900; %sets width and height of the graph
height = 500;
set(gcf,'position',[10,10,width,height])
saveas(gcf,'error_phase2','epsc')
hold off;

figure(11);
hold on;
p = bar(Date(258:265), err_3_comb);
set(p, {'DisplayName'}, {'Adjusted Model Forecasting Errors','Unadjusted Model Forecasting Errors'}')
ax = gca;
ax.FontSize = 12;
title('Max Daily Forecasting Errors in Phase 3', 'FontSize', 16);
xlabel('Date', 'FontSize', 14);
ylabel('Percentage Error', 'FontSize', 14);
ytickformat('%.0f');
grid on;
legend ('show', 'FontSize', 14);
width = 900; %sets width and height of the graph
height = 500;
set(gcf,'position',[10,10,width,height])
saveas(gcf,'error_phase3','epsc')
hold off;

figure(12);
hold on;
p = bar(Date(328:335), err_4_comb);
set(p, {'DisplayName'}, {'Adjusted Model Forecasting Errors','Unadjusted Model Forecasting Errors'}')
ax = gca;
ax.FontSize = 12;
title('Max Daily Forecasting Errors in Phase 4', 'FontSize', 16);
xlabel('Date', 'FontSize', 14);
ylabel('Percentage Error', 'FontSize', 14);
ytickformat('%.0f');
grid on;
legend ('show', 'FontSize', 14);
width = 900; %sets width and height of the graph
height = 500;
set(gcf,'position',[10,10,width,height])
saveas(gcf,'error_phase4','epsc')
hold off;
