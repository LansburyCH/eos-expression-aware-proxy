function [out] = BinaryFile_Read(fn)
% read a binary file in the format of: num_dim(uint64) [dims(uint64)] data_type_byte_num(uint64) [data_type(char)] [data]
% and reshape the output according to dims
% 
%   [out] = BinaryFile_Read(fn)
% 
%   ----- INPUT -----
%   fn: path to the binary file
% 
%   ----- OUTPUT -----
%   out: data in matrix form

fid = fopen(fn, 'r');
num_dim = fread(fid, 1, 'uint64');
dims = fread(fid, num_dim, 'uint64');
data_type_byte_num = fread(fid, 1, 'uint64');
data_type = char(fread(fid, data_type_byte_num, 'char')');

if strcmp(data_type, 'double') || strcmp(data_type, 'd')
    out = fread(fid, prod(dims), 'double');
elseif strcmp(data_type, 'float') || strcmp(data_type, 'f')    
    out = fread(fid, prod(dims), 'single');
elseif strcmp(data_type, 'logical')
    out = logical(fread(fid, prod(dims), 'logical'));
else
    fclose(fid);
    error('Unrecognized data type');
end

fclose(fid);

out = reshape(out, dims');

