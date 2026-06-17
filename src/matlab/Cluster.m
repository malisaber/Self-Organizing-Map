close all
clear
clc


% video writer
filename = 'outputs/animations/k-nearest-neighbors.gif';
if ~exist('outputs/animations', 'dir')
	mkdir('outputs/animations');
end


%initialization
%	data
X( 1:20, :) = (1.5 * rand(20,2)+[ 2  2])-[0.5 0.5];
X(21:40, :) = (1.5 * rand(20,2)+[ 2 -2])-[0.5 0.5];
X(41:60, :) = (1.5 * rand(20,2)+[-2  2])-[0.5 0.5];
X(61:80, :) = (1.5 * rand(20,2)+[-2 -2])-[0.5 0.5];

%initialization
%	neurons
num_neurons = 5;
W = zeros(num_neurons, 2); %(rand(num_neurons,2)-0.5);
WOld = W;

%initialization
%	parameters
alpha = 1e-2;
epsilon = 1e-5;
cntr = 0;
delta_w = 2;
K = 2;
next_increase = 500;
delay = 0.1;

%visualization
f1 = figure;
hold on
px = plot(X(:, 1), X(:, 2), '*', 'MarkerSize',5, 'Color', 'r');
axis([-4 4 -4 4])
pw = plot(W(:, 1), W(:, 2), 'o', 'MarkerSize',6, 'MarkerFaceColor', 'b');
title(sprintf('Agent Movement Using K-Nearest Neighbor Rule (K = %d)', K))
axis off
drawnow;
pause(1);


%simulation
% one epochs
while (cntr < 1500)
	cntr = cntr + 1;
	i = randi(size(X,1));
	delta = W - X(i,:);
	dist = sum(delta .* delta , 2);
	[~, nidx] = mink(dist, K);
	[~, midx] = min (dist);
	W(nidx, :) = W(nidx, :) + (alpha/2) * (X(i,:) - W(nidx, :));
	W(midx, :) = W(midx, :) + (alpha/2) * (X(i,:) - W(midx, :));
	if (mod(cntr, 5) == 1)
		pw.XData = W(:, 1);
		pw.YData = W(:, 2);
		title(sprintf('Agent Movement Using K-Nearest Neighbor Rule (K = %d)', K))
		% plotting the trajectory
		for i=1:num_neurons
			plot([WOld(i, 1), W(i, 1)], [WOld(i, 2), W(i, 2)])
		end
		drawnow;
		WOld = W;
		
		% generating the .gif file
		frame = getframe(gcf);
		img = frame2im(frame);
		[A,map] = rgb2ind(img,256);
		delay = delay * 0.95;
		if delay < 0.03
			delay = 0.03;
		end
		if cntr == 1
			imwrite(A,map,filename,...
				'gif','LoopCount',Inf,...
				'DelayTime',delay);
		else
    	    imwrite(A,map,filename,...
    	        'gif','WriteMode','append',...
    	        'DelayTime',delay);
		end
	end
	if(cntr == next_increase)
		K = K - 1;
		next_increase = next_increase * 2;
		if K <= 1
			K = 1;
			next_increase = 0;
		end
		% disp(K)
	end
	alpha = alpha * 1;
end
disp("simultion is Done after " + cntr + " episodes");
disp("The result saved at """ + filename + """");


