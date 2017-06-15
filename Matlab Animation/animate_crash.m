clc;    % Clear the command window.
close all;  % Close all figures 
clear;  % Erase all existing variables. 

trials = csvread('../rl_logs/run6/trials.csv',1,0);
crashes = csvread('../rl_logs/run5/crashes.csv',1,0);

filename = 'sampleCrash.gif';

crash_instance = struct(...
    'trial', 0,...
    'car_x', zeros(1,1),...
    'car_y', zeros(1,1),...
    'ped_x', zeros(1,1),...
    'ped_y', zeros(1,1),...
    'total_reward', 0,...
    'total_m_d', 0,...
    'steps', 0);

crash_instances(1) = crash_instance;

j = 1;
for i =1:size(crashes,1)
    step = int16(crashes(i,2)) + 1;
    crash_instances(j).car_x(step) = crashes(i,4);
    crash_instances(j).car_y(step) = crashes(i,5);
    crash_instances(j).ped_x(step) = crashes(i,6);
    crash_instances(j).ped_y(step) = crashes(i,7);
    crash_instances(j).total_reward = crash_instances(j).total_reward + crashes(i,10);
    if i == size(crashes,1)
        break
    end
    if int16(crashes(i,1)) ~= int16(crashes(i+1,1))
        crash_instances(j).car_x(step+1) = crashes(i,12);
        crash_instances(j).car_y(step+1) = crashes(i,13);
        crash_instances(j).ped_x(step+1) = crashes(i,14);
        crash_instances(j).ped_y(step+1) = crashes(i,15);
        crash_instances(j).total_m_d = exp(-crash_instances(j).total_reward);
        crash_instances(j).steps = step;
        crash_instances(j).trial = int16(crashes(i,1));
        j = j + 1;
        crash_instances(j) = crash_instance;
    end
end
crash_instances(j).car_x(step+1) = crashes(i,12);
crash_instances(j).car_y(step+1) = crashes(i,13);
crash_instances(j).ped_x(step+1) = crashes(i,14);
crash_instances(j).ped_y(step+1) = crashes(i,15);
crash_instances(j).total_m_d = exp(-crash_instances(j).total_reward);
crash_instances(j).steps = step;
crash_instances(j).trial = int16(crashes(i,1));

%%
trial = 1:length(crash_instances);
figure
plot(trial, cell2mat({crash_instances(trial).total_reward}))
title('total reward vs trial #')
xlabel('trial #')
ylabel('total reward')
grid on
%%
figure
plot(trial, cell2mat({crash_instances(trial).total_m_d}))
title('Mahalanobis Distance vs trial #')
xlabel('trial #')
ylabel('Mahalanobis Distance')
grid on

%%
figure
hold on;
for i = 1:50:492
    plot(cell2mat({crash_instances(i).ped_x}), ...
        cell2mat({crash_instances(i).ped_y}), ...
        'DisplayName', sprintf('Trial %d', i));
end
title('Path Evolutions')
xlabel('<-- X -->')
ylabel('<-- Y -->')
legend('show')
hold off

%%
figure
hold on
trial =490;
plot(cell2mat({crash_instances(trial).ped_x}), ...
        cell2mat({crash_instances(trial).ped_y}));
plot(cell2mat({crash_instances(trial).car_x}), ...
        cell2mat({crash_instances(trial).car_y}))
legend('pedestrian', 'car')
title(sprintf('Trial %d --- Reward: %2.3f | M-Dist: %2.3f',...
    trial,...
    crash_instances(trial).total_reward,...
    crash_instances(trial).total_m_d))
xlabel('<-- X -->')
ylabel('<-- Y -->')
hold off
axis([0 9 0 99])
% x = driveTracking(:,2);
% y = driveTracking(:,3);
% v = driveTracking(:,4);
% xped = driveTracking(:,5);
% yped = driveTracking(:,6);
% crash = driveTracking(:,7);
% 
% f = figure;
% set(f, 'doublebuffer', 'on'); % For smoothness in the animation
%  %[make calls to set up plot axes or do first plot]
%  axis([-10,10,-99,10])
% hold on;
% crashYet=0;
% 
% for i=2:length(x)
%    pause(0.5) % seconds. Adjust to desired speed
%    if crash(i)==0
%     plot(xped(2:i),-yped(2:i),'o--g')
%    else
% %     if crashYet == 0
% %         plot(xped(2:i),-yped(2:i),'o--g') 
% %         crashYet=1;
% %     else
%     plot(xped(2:i),-yped(2:i),'o--r') 
% %     end
%    end
%    
%    plot(0,y(i) - yped(i-1),'^ k','MarkerSize', 10)
%    legend('Pedestrian', 'Car')
%    drawnow;
%    
%    frame = getframe(1);
%    im = frame2im(frame);
%    [imind,cm] = rgb2ind(im,256);
%    if i == 2;
%        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
%    else
%        imwrite(imind,cm,filename,'gif','WriteMode','append');
%    end
%    
% end

