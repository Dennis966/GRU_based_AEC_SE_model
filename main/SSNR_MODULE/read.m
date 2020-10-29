function [output_data, SampleRate]=read(filename)
[y, Fs]=audioread(filename);
output_data=y;
SampleRate=Fs;
end
