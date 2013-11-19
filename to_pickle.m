load('deep.mat')
% features: uid, long, lat, weekday, hour, timestamp, tag1, …, tag825
tagged = sum(A(:, 7:end), 2) > 1;
X = A(tagged, 7:end);
label = 4*(A(tagged, 4)<5) + floor(A(tagged, 5)/6);
[N, d] = size(X);
K = length(unique(label));
class_freq = zeros(K, 1);
class_index = zeros(K, 1);
S = 10000;
test_set = zeros(S, d);
test_label = zeros(S, 1);
val_set = zeros(S, d);
val_label = zeros(S, 1);
train_set = zeros(N-2*S, d);
train_label = zeros(N-2*S, 1);
l_test = 1;
l_val = 1;
l_train = 1;
for k = 1:K
	l = 1;
	tmp = find(label == k-1);
	z = randperm(length(tmp));
	class_index = tmp(z);
	class_freq = length(tmp)/N;

	high = floor(length(class_index)*S/N);
	index = class_index(l:high);
	test_set(l_test:length(index)+l_test-1, :) = X(index, :);
	test_label(l_test:length(index)+l_test-1, :) = label(index);
	l_test = length(index)+l_test+1;

	l = high+1;
	high = floor(length(class_index)*2*S/N);
	index = class_index(l:high);
	val_set(l_val:length(index)+l_val-1, :) = X(index, :);
	val_label(l_val:length(index)+l_val-1, :) = label(index);
	l_val = length(index)+l_val+1;

	l = high+1;
	high = length(class_index);
	index = class_index(l:high);
	train_set(l_train:length(index)+l_train-1, :) = X(index, :);
	train_label(l_train:length(index)+l_train-1, :) = label(index);
	l_train = length(index)+l_train+1;
end
z = randperm(S);
test_set = double(sparse(test_set(z, :)));
test_label = int32(test_label(z, :));
z = randperm(S);
val_set = double(sparse(val_set(z, :)));
val_label = int32(val_label(z, :));
z = randperm(N-2*S);
train_set = double(sparse(train_set(z, :)));
train_label = int32(train_label(z, :));
save('-v7', 'small.mat', 'train_label', 'train_set', 'val_label', 'val_set', 'test_label', 'test_set')
