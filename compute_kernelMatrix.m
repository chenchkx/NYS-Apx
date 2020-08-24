%% compute kernel matrix between two feature matrices with Gaussian, Exponential, Polynomial kernel ,etc.
%  Written by Kai-Xuan Chen (kaixuan_chen_jsh@163.com). If you find any bugs, please contact me.
% 
%  input:
%       fea_a,fea_b : columns of vectors of data points. 
%       options     : Struct value in Matlab.
%               options.type_kernel  -  Choices are:
%                   'Gau'      - Gaussian kernel:      e^{-(|x-y|^2)/2t^2}   
%                   'Exp'      - Exponential kernel:   e^{-(|x-y|)/2t^2}  
%                   'Pol'      - Polynomial kernel:    (x'*y+1)^d    
%                   'Lin'      - Linear kernel:        x'*y    
%
%                options.t       -  parameter for Gaussian, Exponential
%                options.d       -  parameter for Poly
% output:
%       kernel_matrix: kernel matrix between two feature matrices
% 
% 
% If you find this code useful for your research, we appreciate it very much if you can cite our related works:
% 1.
% Chen K X, Wu X J, Ren J Y, et al. More About Covariance Descriptors for Image Set Coding: Log-Euclidean Framework based Kernel Matrix 
% Representation[C]//Proceedings of the IEEE International Conference on Computer Vision Workshops. 2019: 0-0.
% 2.
% Chen K X, Wu X J, Wang R, et al. Riemannian kernel based Nystr?m method for approximate infinite-dimensional covariance descriptors 
% with application to image set classification[C]//2018 24th International conference on pattern recognition (ICPR). IEEE, 2018: 651-656.
%



function kernel_matrix = compute_kernelMatrix(fea_a, fea_b, options)

    if (~exist('options','var'))
       options = [];
    else
       if ~isstruct(options) 
           error('parameter error!');
       end
    end
    if ~isfield(options,'type_kernel')
        options.type_kernel = 'Gau';
    end
    

    switch options.type_kernel
        case 'Gau' 
%           Gaussian kernel:      e^{-(|x-y|^2)/2t^2}            
            if ~isfield(options,'t')
                options.t = 0.2;
            end
            dis_matrix = compute_eudDist(fea_a, fea_b);    
            K = exp(-dis_matrix.^2/(2*options.t^2));           
  
        case 'Exp'        
%           Exponential kernel:   e^{-(|x-y|)/2t^2}  
            if ~isfield(options,'t')
                options.t = 0.2;
            end
            dis_matrix = compute_eudDist(fea_a, fea_b);
            K = exp(-dis_matrix/(2*options.t^2));            

        case 'Pol'     
%           Polynomial kernel:    (x'*y+1)^d    
            if ~isfield(options,'d')
                options.t = 2;
            end
            inner_matrix = fea_a'*fea_b;
            K = (inner_matrix+1).^options.d;          

        case 'Lin'     
%           Linear kernel:        x'*y    
            inner_matrix = fea_a'*fea_b;
            K = inner_matrix;
                      
        otherwise
            error('KernelType does not exist!');
    end
    kernel_matrix = K;
end


function dis_matrix = compute_eudDist(fea_a, fea_b)
    
    aa = sum(fea_a.*fea_a, 1);
    bb = sum(fea_b.*fea_b, 1);
    ab = fea_a'*fea_b;
    temp_d = bsxfun(@plus,aa',bb) - 2*ab;
    temp_d(temp_d<0) = 0;
    dis_matrix = sqrt(temp_d);
    if isequal(fea_a, fea_b)
        dis_matrix = max(dis_matrix, dis_matrix');
    end

end

