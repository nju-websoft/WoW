db="sift1m"
m=16
efc=128
k=10
c=5000
basevec="../exp/data/vecs/sift1m/sift_base.fvecs"
query_vec="../exp/data/vecs/sift1m/sift_query.fvecs"
base_att="../exp/data/meta/meta_c"$c"_n1000000.bin"
# base_att="serial"
query_filter="../exp/data/ranges/common_ranges_c"$c"_nq10000.bin"
gt_file="./common_"$db"_gt_c"$c"_nq10000.bin"
space="l2"


# db="gist1m"
# m=16
# efc=256
# k=10
# c=5000
# basevec="../exp/data/vecs/gist1m/gist_base.fvecs"
# query_vec="../exp/data/vecs/gist1m/gist_query.fvecs"
# base_att="../exp/data/meta/meta_c"$c"_n1000000.bin"
# # base_att="serial"
# query_filter="../exp/data/ranges/common_ranges_c"$c"_nq10000.bin"
# gt_file="./common_gist1m_gt_c"$c"_nq1000.bin"
# space="l2"

# db="arxiv2m"
# m=16
# efc=256
# k=10
# c=100
# basevec="../exp/data/vecs/arxiv2m/arxiv2m_base.fvecs"
# query_vec="../exp/data/vecs/arxiv2m/arxiv2m_query.fvecs"
# base_att="../exp/data/meta/meta_c"$c"_n2138591.bin"
# # base_att="serial"
# query_filter="../exp/data/ranges/common_ranges_c"$c"_nq10000.bin"
# gt_file="./common_arxiv2m_gt_c"$c"_nq10000.bin"
# space="ip"

cd ../build
cmake .. && make -j8
cd ../example
# if not exists ground truth, create it
if [ ! -f $gt_file ]; then
    echo "Creating ground truth file"
    ../build/bin/gengt \
        --k $k \
        --basevec $basevec \
        --queryvec $query_vec \
        --query_rng $query_filter \
        --gt_file $gt_file \
        --att_file $base_att \
        --space $space
fi

../build/bin/searchwow \
    --k $k \
    --query_vec $query_vec \
    --query_rng $query_filter \
    --gt_file $gt_file \
    --space $space \
    --index_location $index
