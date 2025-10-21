% Add an arc to the figure
fig = openfig('WLISmap.fig');
ax = findall(fig, 'Type', 'axes');
ax = ax(1);
hold(ax, 'on');

% xlim = [-0.0148, 0.0148]
% ylim = [0.7109, 0.7243]
xc = 0;
yc = 0.717;

r = 0.0034;
theta1 = 0;
theta2 = 13;

theta = linspace(deg2rad(theta1), deg2rad(theta2), 100);
x = xc + r * cos(theta);
y = yc + r * sin(theta);

plot(ax, x, y, 'y', 'LineWidth', 1);