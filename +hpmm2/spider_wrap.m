
classdef spider_wrap

properties
    % Parameters
    algo = [];
    
    ind_o = [];
    ind_u = [];
    K = [];
end

methods
    % constructor
    function obj = spider_wrap(varargin)               
        obj = set(obj,varargin{:});
    end
    
    function [ obj ] = set( obj, varargin )
        if rem(nargin-1,2)
            error('Arguments to set should be (property, value) pairs')
        end
        numSettings	= (nargin-1)/2;
        for n = 1:numSettings
            property	= varargin{(n-1)*2+1};
            value		= varargin{(n-1)*2+2};
            if isempty(obj.algo) || isprop(obj, property)
                obj.(property) = value;
            else
                % is it is not a property of the current object, then set it as a property
                % of the algo
                obj.algo = set(obj.algo, property, value);
            end
        end
    end
    
    function [d, obj] = training( obj, d )
        % obj = training( obj, X_train, K, X_validation)
        [x, y] = get_xy(d);
        X_train = [x y];
        X_validation = [];
        obj.algo = training( obj.algo, X_train, obj.K, X_validation);
    end
    
    function d = testing( obj, d )
        % X_u = testing( obj, X_o, ind_o, ind_u )
        X_o = get_x(d);
        X_u = testing( obj.algo, X_o, obj.ind_o, obj.ind_u );
        d = set_x(d, X_u);
    end
    
end % methods
end % classdef
