close all
clear
clc


filename = 'outputs/animations/SOM.gif';
if ~exist('outputs/animations', 'dir')
	mkdir('outputs/animations');
end


%initialization
%	data
t1 = (-4 : 4)/5;
t2 = ones(9,1);
x = (t2 * t1);
y = x';
X(:, 1) = x(:);
X(:, 2) = y(:);

%initialization
%	neurons
W = (rand(size(X))-0.5);


%initialization
%	parameters
alpha = 0.1;
epsilon = 1e-5;
cntr = 0;
delta_w = 2;
neuron_cnt = size(X, 1);
K = ceil(1.5 * log2(neuron_cnt));
next_increase = 100;
increase_interval = 50;
pick_count = zeros(neuron_cnt,1);
delay = 0.03;
reallocating_absolete_neurons = false;


%visualization setup
f1 = figure;
hold on
px = plot(X(:, 1), X(:, 2), '*', 'MarkerSize',5);
axis([-1 1 -1 1])
pw = plot(W(:, 1), W(:, 2), 'o', 'MarkerSize',5);
title(sprintf('Agent Movement Using K-Nearest Neighbor Rule (K = %d)', K))
axis off


%simulation
% one epochs
while cntr < 10000
	cntr = cntr + 1;
	i = randi(size(X,1));
	delta = W - X(i,:);
	dist = sum(delta .* delta , 2);
	[~, nidx] = mink(dist, K);
	[~, midx] = min (dist);
	W(nidx, :) = W(nidx, :) + (alpha/2) * (X(i,:) - W(nidx, :));
	W(midx, :) = W(midx, :) + (alpha/2) * (X(i,:) - W(midx, :));
	pick_count(nidx) = pick_count(nidx) + 1;
	
	%visualization
	if (mod(cntr, 25) == 1)
		pw.XData = W(:, 1);
		pw.YData = W(:, 2);
		title(sprintf('Agent Movement Using K-Nearest Neighbor Rule (K = %d)', K))
		drawnow;
		frame = getframe(gcf);
		img = frame2im(frame);
		[A,map] = rgb2ind(img,256);
		%delay = delay * 0.95;
		%if delay < 0.03
		%	delay = 0.03;
		%end
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
	
	%neighbourhood update
	if K > 1
		if(cntr == next_increase)
			increase_interval = ceil(increase_interval * 1.5);
			next_increase = cntr + increase_interval;
			K = K - 1;
			% next_increase = next_increase * 2;
			% disp(K)
		end
	else
		K = 1;
	end
	
	%learning rate update
	alpha = alpha * 0.99999;
	
	%obsolete neurons re-allocation
	if reallocating_absolete_neurons
		if (mod(cntr, 10 * neuron_cnt) == 0)
			idx = find(pick_count < (sum(pick_count,'all')/neuron_cnt/20));
			W(idx, :) = (rand(size(idx,1),2)-0.5)*2;
			pick_count = zeros(neuron_cnt,1);
		end
	end
end

disp("simultion is Done after " + cntr + " episodes");
disp("The result saved at """ + filename + """");
