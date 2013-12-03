function baseline(method, j, seed, c)
j = j+1;
%if j == 4
%j = 3;
%end
c = c+1;
nfold = 4;
assert(and(j>=1, j<=nfold), 'invalid fold')
load('deep.mat')
% features: uid, long, lat, weekday, hour, timestamp, tag1, …, tag825
tagged = sum(A(:, 7:end), 2) > 1;
X = A(tagged, 7:end);
label = 4*(A(tagged, 4) < 5) + ceil(A(tagged, 5)/6) + 1;
K = length(unique(label));
rng(seed)
z = randperm(size(X, 1));
pred = {};
proba = {};
report = zeros(nfold, 6);
allC = logspace(0.6, 1.8, 4);
%allC = [3.5 3.5 30 30];
gammas = reshape(repmat(logspace(-.2,.4,8), 1, 2), 4, 4);
C = allC(c);
C=4.02;
g=0.55;
%for j = 1:nfold
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
        case 'svm'
            %for g = logspace(-.64, .40, 4)
            %for g = gammas(:,c)'
                opt = sprintf('-h 0 -m 5000 -c %.4f -g %.9f -q', C, g);
                model = svmtrain2(double(label(tr)), full(X(tr, :)), opt);
                [pred{j}, acc, ~] = svmpredict(double(label(vl)), full(X(vl, :)), model, '-q');
                fprintf('C: %.4f, g: %.9f, %.5f\n', C, g, acc(1))
            %end
% opt = optimoptions(@quadprog, 'Algorithm', 'interior-point-convex');
% svms = svmtrain(X(vl, :), label(vl), 'boxconstraint', 1.5, 'kernel_function', 'rbf', 'method', 'QP', 'options', opt);
	end
	tt = toc;
	C = confusionmat(gold_label, pred{j});
	report(j, [1 2]) = [sum(diag(C))/length(gold_label) tt];
	res = report(j, [1 2]);
    save(strcat(method, 'f_', int2str(j)), 'res')
    fprintf('DONE with fold %d\n', j)
%end
%report
%disp(100*(1-mean(report(:,1))))
