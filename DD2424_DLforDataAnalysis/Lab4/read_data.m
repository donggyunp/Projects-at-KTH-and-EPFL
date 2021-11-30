%read data
book_fname = 'goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);

book_char = unique(book_data);
K = numel(book_char);

keySet={};
valueSet={};
for i = 1:K
    keySet{i} = book_char(i);
    valueSet{i} = i;
end

char_to_ind = containers.Map(keySet,valueSet);
ind_to_char = containers.Map(valueSet,keySet);