# c is the number of attribute values
db="sift1m"
m=16
efc=128
c=10000
base_vec="../exp/data/vecs/sift1m/sift_base.fvecs"
base_att="../exp/data/meta/meta_c"$c"_n1000000.bin"
space="l2"
index=$db"_"$m"_"$efc"_"$space"_c"$c".wow"
o=4

# db="gist1m"
# m=16
# efc=256
# c=1000
# base_vec="../exp/data/vecs/gist1m/gist_base.fvecs"
# base_att="../exp/data/meta/meta_c"$c"_n1000000.bin"
# space="l2"
# index=$db"_"$m"_"$efc"_"$space"_c"$c".wow"
# o=4

t=16

cd ../build
cmake .. && make -j8
cd ../example

../build/bin/buildwow \
    --m $m \
    --o $o \
    --efc $efc \
    --baseatt $base_att \
    --basevec $base_vec \
    --space $space \
    --index_location $index \
    --threads $t \