db="sift1m"
m=16
efc=128
base_vec="../exp/data/vecs/sift1m/sift_base.fvecs"
base_att="serial"
space="l2"
index=$db"_"$m"_"$efc"_"$space".wow"

t=16

cd ../build
cmake .. && make -j8
cd ../example

../build/bin/buildwow \
    --m $m \
    --efc $efc \
    --baseatt $base_att \
    --basevec $base_vec \
    --space $space \
    --index_location $index \
    --threads $t \