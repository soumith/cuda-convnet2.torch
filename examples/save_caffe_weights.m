function save_caffe_weights(prototxtfilename,modelfilename)
% Save the weigths from a caffe model defined by a proto file and the 
% model file to a Matlab file, so that it can be read by Torch.
% 
% NOTE: Caffe should be added to the path before calling this function.
%       addpath('PATHTOCAFFE/matlab/caffe')
% 
% INPUTS
%  prototxtfilename  - path to the proto file
%  modelfilename     - path to the model file
%
% Copyright (c) 2014 by 
%    Francisco Massa <francisco-vitor.suzano-massa@imagine.enpc.fr>
%    Sergey Zagoruyko <sergey.zagoruyko@imagine.enpc.fr>
% Universite Paris-Est Marne-la-Vallee/ENPC, LIGM, IMAGINE group

[~,weightsfilename,~] = fileparts(modelfilename);

caffe('init', prototxtfilename, modelfilename);

layers = caffe('get_weights');
weights = struct();
for i=1:length(layers),
    weights.([layers(i).layer_names,'_w']) = layers(i).weights{1};
    weights.([layers(i).layer_names,'_b']) = layers(i).weights{2};
end

save([weightsfilename,'_weights.mat'],'-struct','weights');

end
