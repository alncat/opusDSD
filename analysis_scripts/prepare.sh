starname=$(basename $1)
dirn=$(dirname $1)
echo $dirn $starname
filename=$(basename $starname)
dsd parse_pose_star $1 -D $2 --Apix $3 -o $dirn/$filename\_pose_euler.pkl $4
dsd parse_ctf_star $1 -D $2 --Apix $3 -o $dirn/$filename\_ctf.pkl $4
