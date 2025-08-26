db="sift1m"
m=16
efc=128
k=10
basevec="../exp/data/vecs/sift1m/sift_base.fvecs"
query_vec="../exp/data/vecs/sift1m/sift_query.fvecs"
query_filter_dir="../exp/data/ranges/intrng10k_1000000/"
gt_dir="../exp/data/gt/gt_sift_k"$k"/"
space="l2"
index=$db"_"$m"_"$efc"_"$space".wow"

cd ../build
cmake .. && make -j8
cd ../example

rng_list=(17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0)
rng_list=(17)

for i in ${rng_list[@]}
do
    echo "Running with range $i"
    rng_file=$query_filter_dir$i".bin"
    gt_file=$gt_dir$i".bin"

    if [ ! -f $gt_file ]; then
    echo "Creating ground truth file"
    ../build/bin/gengt \
        --k $k \
        --basevec $basevec \
        --queryvec $query_vec \
        --query_rng $rng_file \
        --gt_file $gt_file \
        --att_file "serial" \
        --space $space
    fi

    ../build/bin/searchwow \
        --k $k \
        --query_vec $query_vec \
        --query_rng $rng_file \
        --gt_file $gt_file \
        --space $space \
        --index_location $index
done