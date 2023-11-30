python -m cryodrgn.commands.parse_pose_star $1/$2.star -D $3 --Apix $4 -o $1/$2_pose_euler.pkl $5
python -m cryodrgn.commands.parse_ctf_star $1/$2.star -D $3 --Apix $4 -o $1/$2_ctf.pkl $5
