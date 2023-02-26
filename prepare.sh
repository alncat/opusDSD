python -m cryodrgn.commands.parse_pose_star $1/$2.star -D $3 --Apix $4 -o $1/$2_pose_euler.pkl #--relion31
python -m cryodrgn.commands.parse_ctf_star $1/$2.star -D $3 --Apix $4 -o $1/$2_ctf.pkl -o-g $1/$2_grp.pkl --ps 0 #--relion31 --kv 300 --cs 2 -w 0.1
