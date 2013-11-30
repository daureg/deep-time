clear all;
method='knn';
nfold = 4;
load('deep.mat')
% features: uid, long, lat, weekday, hour, timestamp, tag1, …, tag825
tagged = sum(A(:, 7:end), 2) > 1;
X = A(tagged, 7:end);
label = 4*(A(tagged, 4) < 5) + ceil(A(tagged, 5)/8) + 1;
K = length(unique(label));
z = randperm(size(X, 1));
pred = {};
proba = {};
report = zeros(nfold, 6);
for j = 1:nfold
	j
	[tr, vl] = get_cross_set(z, nfold, j);
	gold_label = label(vl);
	tic;
	switch method
		case 'nb'
			mlnb = NaiveBayes.fit(X(tr, :), label(tr), 'dist', 'mn');
			[proba{j}, pred{j}] = mlnb.posterior(X(vl, :));
		case 'knn'
			mdl = ClassificationKNN.fit(X(tr,:),label(tr),'Distance', 'hamming', 'NumNeighbors', 5);
			% not a probabistic method, the proba is the number of k
			% nearest neighbors that have the selected class.
			[pred{j}, proba{j}] = predict(mdl, X(vl,:));
	end
	tt = toc;
	C = confusionmat(gold_label, pred{j});
	report(j, [1 2]) = [sum(diag(C))/length(gold_label) tt];
end
report
disp(100*(1-mean(report(:,1))))
