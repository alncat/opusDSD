starname=$(basename $1)
dirn=$(dirname $1)
filename=$(basename $starname .star)
echo $dirn $filename
python -m cryodrgn.commands.parse_multi_pose_star $1 -D $2 --Apix $3 -o $dirn/$filename\_pose_euler.pkl --masks $4 --bodies $5 $6 $7 $8
python -m cryodrgn.commands.parse_ctf_star $1 -D $2 --Apix $3 -o $dirn/$filename\_ctf.pkl $8
