function [] = BinaryFile_Write(fn, data)
% write a binary file in the format of: num_dim(uint64) [dims(uint64)] data_type_byte_num(uint64) [data_type(char)] [data]
% 
%   [] = BinaryFile_Read(data, fn)
% 
%   ----- INPUT -----
%   fn: 
%       path to the binary file
%   data:
%       data to be saved

dims = size(data);
num_dim = length(dims);
data_info = whos('data');
data_type = data_info.class;
data_type_byte_num = length(data_type);

fid = fopen(fn, 'w');
fwrite(fid, num_dim, 'uint64');
fwrite(fid, dims, 'uint64');
fwrite(fid, data_type_byte_num, 'uint64');
fwrite(fid, data_type, 'char');
fwrite(fid, data, data_type);

fclose(fid);

