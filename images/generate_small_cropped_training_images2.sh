#!/bin/bash

# first argument - dir with big images
# second - dir with result images
# third - start index
# fouth - count of iters per image

out_dir=$2
let i=$3
let count=$4
temp_dir='temp/'
declare -A AREAMAP=( [bear.jpg]='0.4' [crocodile.jpg]='0.3' [dog.jpg]='0.3' [horse2.jpg]='0.5' [kangoro.jpg]='0.5' [rabbit.jpg]='0.45' [sheep.jpg]='0.45' [walrus.jpg]='0.5' )
declare -A LEMAP=( [bear.jpg]='0.3' [crocodile.jpg]='0' [dog.jpg]='0.3' [horse2.jpg]='0' [kangoro.jpg]='0.3' [rabbit.jpg]='0.2' [sheep.jpg]='0.1' [walrus.jpg]='0.25' )
declare -A RIMAP=( [bear.jpg]='0' [crocodile.jpg]='0.1' [dog.jpg]='0.15' [horse2.jpg]='0.1' [kangoro.jpg]='0' [rabbit.jpg]='0.1' [sheep.jpg]='0.2' [walrus.jpg]='0.1' )
declare -A UPMAP=( [bear.jpg]='0' [crocodile.jpg]='0.25' [dog.jpg]='0.1' [horse2.jpg]='0' [kangoro.jpg]='0' [rabbit.jpg]='0' [sheep.jpg]='0' [walrus.jpg]='0' )
declare -A BOMAP=( [bear.jpg]='0.25' [crocodile.jpg]='0.3' [dog.jpg]='0.1' [horse2.jpg]='0.3' [kangoro.jpg]='0' [rabbit.jpg]='0.1' [sheep.jpg]='0.2' [walrus.jpg]='0.1' )
mkdir -p "$temp_dir"
for f in `ls -v "$1"`
do
 if [[ $f != *".jpg"* ]]
 then
  continue
 fi
 #~ if [[ $f == *"white"* ]] || [[ $f == *"bear"* ]] || [[ $f == *"crocodile"* ]] || [[ $f == *"horse"* ]] || [[ $f == *"dog"* ]] || [[ $f == *"kango"* ]]
 #~ then
  #~ continue
 #~ fi
 if [[ $f == *"walrus"* ]]
  then
   continue
 fi
 echo generate_train_croped_images2.py --name "$1$f" --l ${LEMAP[$f]} --r ${RIMAP[$f]} --u ${UPMAP[$f]} --b ${BOMAP[$f]} --area ${AREAMAP[$f]} --start $i --count "$count" --out_dir $temp_dir
 python generate_train_croped_images2.py --name "$1$f" --l ${LEMAP[$f]} --r ${RIMAP[$f]} --u ${UPMAP[$f]} --b ${BOMAP[$f]} --area ${AREAMAP[$f]} --start $i --count "$count" --out_dir $temp_dir
 ((i+=count))
done


echo turn_images.py $temp_dir $temp_dir
python turn_images.py $temp_dir $temp_dir

for f in `ls -v $temp_dir`
do
 echo resize_image_to_cube.py "$temp_dir$f" $out_dir
 python resize_image_to_cube.py "$temp_dir$f" $out_dir
done

rm -rf "$temp_dir"
