classdef NormalizeLp < dagnn.ElementWise
  properties
    p = 1
    epsilon = eps
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnnormalizelp(inputs{1}, [], 'p', obj.p, 'epsilon', obj.epsilon) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      derInputs{1} = vl_nnnormalizelp(inputs{1}, derOutputs{1}, 'p', obj.p, 'epsilon', obj.epsilon) ;
      derParams = {} ;
    end

    function obj = NormalizeLp(varargin)
      obj.load(varargin) ;
    end
  end
end
