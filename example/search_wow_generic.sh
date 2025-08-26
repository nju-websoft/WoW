db="sift1m"
m=16
efc=128
k=10
basevec="../exp/data/vecs/sift1m/sift_base.fvecs"
query_vec="../exp/data/vecs/sift1m/sift_query.fvecs"
gt_dir="../exp/data/gt/gt_sift_k"$k"/"
space="l2"
index=$db"_"$m"_"$efc"_"$space".wow"

cd ../build
cmake .. && make -j8
cd ../example

pass_list=(500000 250000 100000 50000 25000)
pass_list=(100000)

for npass in ${pass_list[@]}
do
    echo "Running with range $i"

    ../build/bin/searchwowgeneric \
        --k $k \
        --npass $npass \
        --base_vec $basevec \
        --query_vec $query_vec \
        --space $space \
        --index_location $index
done