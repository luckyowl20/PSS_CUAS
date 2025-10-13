% CUAS 3D barrel pivot system using 2 linear actuators
clear;

% Paremeters (change these)
b_a = 30; % distance from barrel pivot to actuator attachment points
L_d = 20; % distance between actuator pivot points
b_l = 60; % Barrel length
actuator_height = 30; % vertical distance from actuators and barrel piot

theta = 20; % right/left rotation
phi = 60; % up/down tilt

% Place pivots on frame
pivot = [0, 0, 0]; % barrel pivot goes on orign
mount1 = [0,  L_d/2, actuator_height]; % actuator 1 pivot (blue)
mount2 = [0, -L_d/2, actuator_height]; % actuator 2 pivot (red)

% Get direction vector
dir = [cosd(phi)*cosd(theta), cosd(phi)*sind(theta), sind(phi)];

% Barrel attachment point (where actuators connect)
attachPt = pivot + b_a * dir;
barrelEnd = pivot + b_l * dir;

% Get lengths of actuator vectors
L1 = norm(attachPt - mount1);
L2 = norm(attachPt - mount2);

% Plotting
figure('Color','w'); hold on; grid on; axis equal;

% wall
fill3([0 0 0 0], [-L_d*2 L_d*2 L_d*2 -L_d*2], [-20 -20 60 60], ...
      [0.9 0.9 0.9], 'FaceAlpha', 0.4, 'EdgeColor', 'none');

% Barrel
plot3([pivot(1), barrelEnd(1)], [pivot(2), barrelEnd(2)], [pivot(3), barrelEnd(3)], ...
      'k-', 'LineWidth', 6);

% Actuators
plot3([mount1(1), attachPt(1)], [mount1(2), attachPt(2)], [mount1(3), attachPt(3)], ...
      'b-', 'LineWidth', 2);
plot3([mount2(1), attachPt(1)], [mount2(2), attachPt(2)], [mount2(3), attachPt(3)], ...
      'r-', 'LineWidth', 2);

% Points
scatter3(pivot(1), pivot(2), pivot(3), 80, 'k', 'filled');
scatter3(attachPt(1), attachPt(2), attachPt(3), 80, 'g', 'filled');
scatter3([mount1(1), mount2(1)], [mount1(2), mount2(2)], [mount1(3), mount2(3)], ...
          80, 'filled');

% Labels and view
xlabel('X (cm)'); ylabel('Y (cm)'); zlabel('Z (cm)');
title(sprintf('3D Barrel Pivot System (\\theta = %.1f°, \\phi = %.1f°)', phi, theta));
legend({'CUAV frame', 'Barrel', sprintf('Actuator 1 (%.2f cm)', L1), ...
    sprintf('Actuator 2 (%.2f cm)', L2) ...
    }, 'Location', 'bestoutside');

view(45, 25);
axis([-10 40 -30 30 -10 60]);

disp(['L1 = ', num2str(L1, '%.2f'), ' cm']);
disp(['L2 = ', num2str(L2, '%.2f'), ' cm']);
