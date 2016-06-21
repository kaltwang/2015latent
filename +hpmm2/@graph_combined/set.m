function [ obj ] = set( obj, varargin )
    if rem(nargin-1,2)
      error('Arguments to set should be (property, value) pairs')
    end
    numSettings	= (nargin-1)/2;
    for n = 1:numSettings
      property	= varargin{(n-1)*2+1};
      value		= varargin{(n-1)*2+2};
      obj.(property) = value;
    end
end

