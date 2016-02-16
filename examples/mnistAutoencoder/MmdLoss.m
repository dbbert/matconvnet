classdef MmdLoss < dagnn.Loss
    
  methods      
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnMMDloss(inputs{1}, inputs{2}, [], obj.opts{:}) ;
      n = obj.numAveraged ;
      m = n + 1 ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnMMDloss(inputs{1}, inputs{2}, derOutputs{1}, obj.opts{:}) ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

%     function reset(obj)
%       obj.average = 0 ;
%       obj.numAveraged = 0 ;
%     end

    function obj = MmdLoss(varargin)
      obj.load(varargin) ;
    end
  end
end