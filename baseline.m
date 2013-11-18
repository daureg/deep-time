clear all;
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
	tmp = zeros(K, 6);
	for k = 1:K
		gold_label = label(vl) == k;

		tic;
		mlnb = NaiveBayes.fit(X(tr, :), label(tr) == k, 'dist', 'mn');
		[proba{j}, pred{j}] = mlnb.posterior(X(vl, :));
		ttime = toc;

		[a, p, r, f] = evaluate(pred{j}, gold_label);
		C = confusionmat(gold_label, pred{j});
		fp = C(1,2)/sum(C(1,:));
		tmp(k, :) = [a, p, r, f, fp, ttime];
	end
	report(j, :) = mean(tmp);
end
report
