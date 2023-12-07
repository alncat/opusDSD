starname=$(basename $1)
dirn=$(dirname $1)
echo $dirn $starname
filename=$(basename $starname)
python -m cryodrgn.commands.parse_multi_pose_star $1 -D $2 --Apix $3 -o $dirn/$filename\_pose_euler.pkl --masks $4 --bodies $5 $6 $7
python -m cryodrgn.commands.parse_ctf_star $1 -D $2 --Apix $3 -o $dirn/$filename\_ctf.pkl $7
