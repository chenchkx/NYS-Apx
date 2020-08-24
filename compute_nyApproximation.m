%% An implementation of nystrom method for approximating infinite-dimensional sample in the RKHS
% Written by Kai-Xuan Chen, (e-mail: kaixuan_chen_jsh@163.com)
% 
% input:
%     X: columns of vectors of data points. 
%     num_sampled: the number of sampled data points
%     r: the dimension of the approximate data points
% output
%     X_appro: the approximation of the data points in the kernel space.
% 
% 
% If you find this code useful for your research, we appreciate it very much if you can cite our related works:
% https://github.com/Kai-Xuan/AidCovDs/  
% 
% Chen K X, Wu X J, Wang R, et al. Riemannian kernel based Nystr?m method for approximate infinite-dimensional covariance descriptors 
% with application to image set classification[C]//2018 24th International conference on pattern recognition (ICPR). IEEE, 2018: 651-656.
% 
% BibTex : 
% @inproceedings{chen2018riemannian,
%   title={Riemannian kernel based Nystr{\"o}m method for approximate infinite-dimensional covariance descriptors with application to image set classification},
%   author={Chen, Kai-Xuan and Wu, Xiao-Jun and Wang, Rui and Kittler, Josef},
%   booktitle={2018 24th International Conference on Pattern Recognition (ICPR)},
%   pages={651--656},
%   year={2018},
%   organization={IEEE}
% }




function X_appro = compute_nyApproximation(X,num_sampled,r)

    if r > num_sampled
        error('should sample m (now is %d) columns not less than the rank r(=%d)', num_sampled, r);
    end

    index_rand = randperm(size(X,2)); 
    X_sampled = X(:,index_rand(1:num_sampled));
    options.type_kernel = 'Gau';    % Gaussian kernel selected here, you can replace it by 'Exp', 'Pol', etc.
    kernel_sampled = compute_kernelMatrix(X_sampled, X_sampled, options);  
    kernel_sampAll = compute_kernelMatrix(X_sampled ,X, options);    
    
    
    if r > rank(kernel_sampled)
        r = rank(kernel_sampled);       
    end
    
    [U,S] = compute_svd(kernel_sampled);
    S = diag(S);
    S_minusSquare = S(1:r).^(-0.5);
    S_minusSquare = diag(S_minusSquare);
    U = U(:,1:r);
    X_appro = S_minusSquare'*U'*kernel_sampAll;   

end
