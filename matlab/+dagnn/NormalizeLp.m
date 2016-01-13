classdef NormalizeLp < dagnn.ElementWise
  properties
    p = 1
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnnormalizelp(inputs{1}, [], obj.p) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      derInputs{1} = vl_nnnormalizelp(inputs{1}, derOutputs{1}, obj.p) ;
      derParams = {} ;
    end

    function obj = NormalizeLp(varargin)
      obj.load(varargin) ;
    end
  end
end
