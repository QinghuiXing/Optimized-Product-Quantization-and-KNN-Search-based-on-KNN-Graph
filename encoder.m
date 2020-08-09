% @brief:
%   this is a demo to encode results from OPQ into binary files.
% @parameters:
%   code length =256
%   k = 256 (8bits)
% @dataset: SIFT-1M

function encoder(code_opq_np, centers_table_opq_np, R_opq_np, path_of_query)

% Encoding centroids
centroids = zeros(256,128);
for i = 1:32
    centroids(:,(i-1)*4+1:i*4) = centers_table_opq_np{i};    
end
fileIN_centroids = fopen('centroids.fcode','wb');
fwrite(fileIN_centroids, centroids', 'float');
fclose(fileIN_centroids);

% Encoding codebook
fileIN_codebook = fopen('codebook.u8code','wb');
fwrite(fileIN_codebook, code_opq_np');
fclose(fileIN_codebook);

% Encoding lookuptable
lookup_table = zeros(256,256,32);
fileIN_lookuptable = fopen('lookuptable.fcode','ab');
for i=1:32
    tmp = centers_table_opq_np{i};
    for row = 1:256
        start = tmp(row,:);
        for col=1:256
            dst = tmp(col,:);
            dis = norm(start-dst)^2;
            lookup_table(row,col,i) = dis;
            lookup_table(col,row,i) = dis;
        end
    end
    fwrite(fileIN_lookuptable, lookup_table(:,:,i)', 'float');
    %dlmwrite('lookuptable_256codelength_100iter.txt',lookup_table(:,:,i),'-append','delimiter',' ','precision','%.5f','newline','pc');
end
fclose(fileIN_lookuptable);

% Encoding query
query = fvecs_read(path_of_query);
query = query';
queryR = query * R_opq_np;

fileID_query = fopen('queryR.fcode','wb');
fwrite(fileID_query,queryR','float');
fclose(fileID_query);

end